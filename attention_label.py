import numpy as np
from transformers import BertTokenizer
import pandas as pd
from utils.data_class import Counterfactual
import random
from utils.config import Config
import pickle
import re
import io, time
from itertools import combinations, cycle, product
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pydictionary import Dictionary
from gensim import utils
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from tqdm import tqdm
from tqdm.contrib import tzip
from sklearn import preprocessing
from nltk.corpus import stopwords

def load_pkl(path):
    pickle_file = open(path,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data

def get_snt(token):
    from nltk.corpus import wordnet as wn
    walks = wn.synsets(token)
    ant = [walk_lemma.antonyms() for walk_lemmas in [walk.lemmas() for walk in walks] for walk_lemma in walk_lemmas if
     walk_lemma.antonyms() != []]

    return ant
def save_index_vocab(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    df = ds.train
    vocab_df = ds.antonym_vocab
    sentences = list(df['text'].values.astype('U'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    attention_masks = encoded_inputs['attention_mask']

    index_vocab_sens = []
    for ids, masks in tzip(dataset_id[:], attention_masks[:]):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        # print(tokens)
        index_vocab = []
        for j, token in enumerate(tokens):
            if token in vocab_df.term.values.tolist():  # if this bert token in vocab
                index_vocab.append(j)
        # print(index_vocab)
        index_vocab_sens.append(index_vocab)
    ds.train["index_vocab_sens"] = index_vocab_sens
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def save_cate(data_name,scale):
    ds = load_pkl(Config.DATA_DIC[data_name])
    delta_T = ds.deltaT_sens
    delta_Y = ds.deltaY_sens
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.5, 1))
    att_label = []
    for i,(ts,ys) in enumerate(tzip(delta_T[:],delta_Y[:])):
        if len(ts)==0:
            att_label.append([])
        else:
            if scale:
                ts = [[x] for x in ts]
                ys = [[x] for x in ys]

                ts = min_max_scaler.fit_transform(np.asarray(ts))
                ys = min_max_scaler.fit_transform(np.asarray(ys))

                # print(np.squeeze(ts))
                ts = np.squeeze(ts,axis=-1)
                ys = np.squeeze(ys,axis=-1)

            labels = []
            for t,y in zip(ts,ys):
                try:
                    labels.append(y/t)
                    # print(y,t,y/t)
                except:
                    print(y,t)
            att_label.append(labels)

    ds.train["cate"] = att_label
    # pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def get_data_glove_syn():
    glove_input_file = 'resource/glove/glove.840B.300d.txt'
    texts = open(glove_input_file, encoding="utf-8").readlines()
    for data_name in ["IMDB-S","IMDB-L","KINDLE"]:
        ds = load_pkl(Config.DATA_DIC[data_name])
        vocab_lst = ds.antonym_vocab.term.values.tolist()
        glove_data_file = f'resource/glove/glove.{data_name}.300d.txt'
        f = open(glove_data_file, "w", encoding='utf-8')
        print(f"selecting the vec from {data_name}")
        for x in tqdm(texts):
            w = x.strip().split()[0]
            if w in vocab_lst:
                f.write(x)
        f.close()
    for data_name in ["IMDB-S", "IMDB-L", "KINDLE"]:
        ds = load_pkl(Config.DATA_DIC[data_name])
        glove_data_file = f'resource/glove/glove.{data_name}.300d.txt'
        glove_model = KeyedVectors.load_word2vec_format(glove_data_file, binary=False, no_header=True)
        synonyms_list = []
        for ri, row in ds.antonym_vocab.iterrows():
            word = row["term"]
            try:
                synonyms_tuple = glove_model.most_similar(word, topn=5)
                synonyms = [w for w, sim in synonyms_tuple if sim >= 0.75]
            except:
                synonyms = []
            synonyms_list.append(synonyms)
        ds.antonym_vocab["synonyms"] = synonyms_list
        pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))

def get_lexicon_wd(data_name):

    pos_words = open("resource/hate lexicon/positive-words.txt", encoding="ISO-8859-1").readlines()
    neg_words = open("resource/hate lexicon/negative-words.txt", encoding="ISO-8859-1").readlines()
    lexicon_lst = pos_words + neg_words
    lexicon_lst = [x.strip() for x in lexicon_lst]
    print(lexicon_lst)
    ds = load_pkl(Config.DATA_DIC[data_name])

    df = ds.train
    sentences = list(df['text'].values.astype('U'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    ds.train["dataset_ids"] = dataset_id

    stop_lst = stopwords.words('english')

    catewd_sens = []

    for ids, cate, index_vocab in tzip(dataset_id[:],
                                      ds.train["cate"].values.tolist()[:],
                                      ds.train["index_vocab_sens"].values.tolist()[:]):
        if len(index_vocab)==0:
            catewd_sens.append([])
        else:
            old_mask_tokens = tokenizer.convert_ids_to_tokens([ids[position] for position in index_vocab])
            catewd_sen = []
            for token in old_mask_tokens:
                if (token in lexicon_lst) and (token not in stop_lst):
                    catewd_sen.append(token)

            catewd_sens.append(catewd_sen)


    ds.train["cate_wds"] = catewd_sens
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def get_top_wd(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    vocab_df = ds.antonym_vocab
    df = ds.train
    top_df = ds.top_terms
    top_lst = top_df.term.values.tolist()
    sentences = list(df['text'].values.astype('U'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    ds.train["dataset_ids"] = dataset_id

    stop_lst = stopwords.words('english')

    catewd_sens = []

    for ids, cate, index_vocab in tzip(dataset_id[:],
                                      ds.train["cate"].values.tolist()[:],
                                      ds.train["index_vocab_sens"].values.tolist()[:]):
        if len(index_vocab)==0:
            catewd_sens.append([])
        else:
            old_mask_tokens = tokenizer.convert_ids_to_tokens([ids[position] for position in index_vocab])
            catewd_sen = []
            for token in old_mask_tokens:
                if (token in top_lst) and (token not in stop_lst):
                    catewd_sen.append(token)

            catewd_sens.append(catewd_sen)


    ds.train["cate_wds"] = catewd_sens
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def get_cate_wd(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    vocab_df = ds.antonym_vocab
    df = ds.train
    sentences = list(df['text'].values.astype('U'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    ds.train["dataset_ids"] = dataset_id

    stop_lst = stopwords.words('english')

    catewd_sens = []

    for ids, cate, index_vocab in tzip(dataset_id[:],
                                      ds.train["cate"].values.tolist()[:],
                                      ds.train["index_vocab_sens"].values.tolist()[:]):
        if len(index_vocab)==0:
            catewd_sens.append([])
        else:
            old_mask_tokens = tokenizer.convert_ids_to_tokens([ids[position] for position in index_vocab])
            # print(old_mask_tokens)
            unstop_tokens_postion = []
            for k, mask_token in enumerate(old_mask_tokens):
                if mask_token in stop_lst:
                    pass
                else:
                    unstop_tokens_postion.append(k)
            # print(unstop_tokens_postion)

            cate = np.array(cate)
            cate = cate[unstop_tokens_postion]
            index_vocab = np.array(index_vocab)
            index_vocab = index_vocab[unstop_tokens_postion]
            index_cate_dic= {}
            for x, y in zip(index_vocab, cate):
                index_cate_dic[x]=y
            sorted_index_cate_dic = sorted(index_cate_dic.items(), key=lambda x: x[1], reverse=True)
            catewd_sen = []
            for (k,v) in sorted_index_cate_dic:
                imp_word = tokenizer.convert_ids_to_tokens(ids[k])
                ant_dic = vocab_df[vocab_df["term"]==imp_word]["antonyms"].values.tolist()[0]

                if ant_dic != "{}":
                    catewd_sen.append(imp_word)
                    syn_wds = vocab_df[vocab_df["term"]==imp_word]["synonyms"].values.tolist()[0]
                    catewd_sen.extend(syn_wds)

                    break
            catewd_sens.append(catewd_sen)


    ds.train["cate_wds"] = catewd_sens
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))

def get_cate_wd_same_coef(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    vocab_df = ds.antonym_vocab
    df = ds.train
    top_df = ds.top_terms
    top_lst = top_df.term.values.tolist()
    sentences = list(df['text'].values.astype('U'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    ds.train["dataset_ids"] = dataset_id

    stop_lst = stopwords.words('english')

    catewd_sens = []

    for ids, cate, index_vocab in tzip(dataset_id[:],
                                      ds.train["cate"].values.tolist()[:],
                                      ds.train["index_vocab_sens"].values.tolist()[:]):
        if len(index_vocab)==0:
            catewd_sens.append([])
        else:
            old_mask_tokens = tokenizer.convert_ids_to_tokens([ids[position] for position in index_vocab])
            # print(old_mask_tokens)
            unstop_tokens_postion = []
            for k, mask_token in enumerate(old_mask_tokens):
                if mask_token in stop_lst:
                    pass
                else:
                    unstop_tokens_postion.append(k)
            # print(unstop_tokens_postion)

            cate = np.array(cate)
            cate = cate[unstop_tokens_postion]
            index_vocab = np.array(index_vocab)
            index_vocab = index_vocab[unstop_tokens_postion]
            index_cate_dic= {}
            for x, y in zip(index_vocab, cate):
                index_cate_dic[x]=y
            sorted_index_cate_dic = sorted(index_cate_dic.items(), key=lambda x: x[1], reverse=True)
            catewd_sen = []
            for (k,v) in sorted_index_cate_dic:
                imp_word = tokenizer.convert_ids_to_tokens(ids[k])
                # ant_dic = vocab_df[vocab_df["term"]==imp_word]["antonyms"].values.tolist()
                ant_lst = get_snt(imp_word)
                # if len(syn_wds)>0:
                #     syn_ant_lst = [vocab_df[vocab_df["term"] == x]["antonyms"].values.tolist()[0] for x in syn_wds]
                #     syn_ant_lst.extend(ant_dic)
                #     ant_dic = list(set(syn_ant_lst))

                # if len(ant_dic)==1 and ant_dic[0] != "{}":
                if len(ant_lst)!=0:
                    syn_wds = vocab_df[vocab_df["term"] == imp_word]["synonyms"].values.tolist()[0]

                    imp_coef = vocab_df[vocab_df["term"] == imp_word]["coef"].values.tolist()[0]
                    catewd_sen.append(imp_word)
                    catewd_sen.extend(syn_wds)
                    new_mask_tokens = tokenizer.convert_ids_to_tokens([ids[position] for position in index_vocab])
                    for token in new_mask_tokens:
                        token_coef = vocab_df[vocab_df["term"] == token]["coef"].values.tolist()[0]
                        if imp_coef*token_coef>0 and (imp_word in top_lst):
                            catewd_sen.append(token)
                    break

            catewd_sens.append(catewd_sen)


    ds.train["cate_wds"] = catewd_sens
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))

def generate_ct_sentences(data_name):
    """
    Generate counterfactual sentences for those contain causal words:
        - substitute all the causal words to antonyms;
        - antonyms: top term with opposite coefficient;
        - If no antonyms, keep the original causal word;
    """
    random.seed(42)
    ds = load_pkl(Config.DATA_DIC[data_name])
    vocab_df = ds.antonym_vocab
    # top_df = ds.top_terms
    # top_lst = top_df.term.values.tolist()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ct_text_cate = []

    for cate_sen,ids in tzip(ds.train["cate_wds"].values.tolist()[:],
                             ds.train["dataset_id"].values.tolist()[:]):
                             # ds.train["index_vocab_sens"].values.tolist()):
        # print(ids)
        # print("cate_sen", cate_sen)
        # break
        tokens = tokenizer.convert_ids_to_tokens(ids)
        # print(tokens)
        # cate_top = [w_cate for token, w_cate in zip(tokens, cate_sen) if token in top_lst]
        if len(cate_sen) > 0:
            new_wds = []
            for token in tokens:
                # print(token)
                if token in cate_sen:# and len(get_snt(token))!=0:
                    sub_w = list(eval(vocab_df[vocab_df['term'] == token].antonyms.values[0]).keys())
                    # print(token,sub_w)
                    if (len(sub_w) == 1):
                        ct_wd = str(sub_w[0])
                    elif (len(sub_w) > 1):
                        ct_wd = str(random.sample(sub_w, 1)[0])
                    else:  # if no antonyms then remove current word
                        ct_wd = ""
                    new_wds.append(ct_wd)
                else:
                    new_wds.append(token)
            if (new_wds == tokens):  # no antonym for the causal word
                ct_text_cate.append(' ')
            else:
                ct_text_cate.append(tokenizer.convert_tokens_to_string(new_wds[1:-1]))
        else:  # no causal words here
            ct_text_cate.append(' ')

    ds.train["ct_text_cate"] = ct_text_cate
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def organize_data(ds, limit=''):
    """
    Organize data for easy use in the evaluation
    'original','original+ctf_by_predicted_causal','original+ctf_by_annotated_causal','original+ctf_by_all_causal','original+ctf_by_human'
    "original+ctf_by_predicted_cate"

    """
    train_data = {}


    if (limit == 'ct'):  # only those have counterfactuals as original data
        ds.train['len_ct_text_causal'] = ds.train['ct_text_causal'].apply(lambda x: len(x.strip()))
        df_select = ds.train[ds.train['len_ct_text_causal'] > 0]
        train_data['original'] = {'text': df_select.text.values, 'label': df_select.label.values}
    else:
        train_data['original'] = {'text': ds.train.text.values, 'label': ds.train.label.values}

    if (ds.moniker == 'imdb'):
        train_data['ctf_by_human'] = {'text': ds.train.ct_text_amt.values, 'label': ds.train.ct_label.values}
        train_data['original+ctf_by_human'] = {
            'text': list(train_data['original']['text']) + list(ds.train.ct_text_amt.values),
            'label': list(train_data['original']['label']) + list(ds.train.ct_label.values)}

    elif (ds.moniker == 'imdb_sents'):
        train_data['ctf_by_human'] = {'text': ds.train_ct.text.values, 'label': ds.train_ct.label.values}
        train_data['original+ctf_by_human'] = {
            'text': list(train_data['original']['text']) + list(ds.train_ct.text.values),
            'label': list(train_data['original']['label']) + list(ds.train_ct.label.values)}

    for flag, col in zip(['predicted_causal', 'annotated_causal', 'all_causal',"cate_causal"],
                         ['identified_causal', 'causal', 'all_causal', "cate"]):
        ds.train['len_ct_text_' + col] = ds.train['ct_text_' + col].apply(lambda x: len(x.strip()))
        df_train_ct_text_flag = ds.train[ds.train['len_ct_text_' + col] > 0]
        train_data['ctf_' + flag] = {'text': df_train_ct_text_flag['ct_text_' + col].values,
                                     'label': df_train_ct_text_flag.ct_label.values}
        train_data['original+ctf_by_' + flag] = {
            'text': list(train_data['original']['text']) + list(df_train_ct_text_flag['ct_text_' + col].values),
            'label': list(train_data['original']['label']) + list(df_train_ct_text_flag.ct_label.values)}

    test_data = {}
    ds.test['len_ct_text_causal'] = ds.test['ct_text_causal'].apply(lambda x: len(x.strip()))
    df_test_ct_text_causal = ds.test[ds.test.len_ct_text_causal > 0]

    test_data['Original'] = {'text': ds.test.text.values, 'label': ds.test.label.values}
    test_data['ct_causal'] = {'text': df_test_ct_text_causal.ct_text_causal.values,
                              'label': df_test_ct_text_causal.ct_label.values}

    if (ds.moniker == 'imdb_sents'):
        test_data['Counterfactual'] = {'text': ds.test_ct.text.values, 'label': ds.test_ct.label.values}
    else:
        test_data['Counterfactual'] = {'text': ds.test.ct_text_amt.values, 'label': ds.test.ct_label.values}

    if (ds.moniker == 'kindle'):
        test_data['original_selected'] = {'text': ds.select_test.text.values, 'label': ds.select_test.label.values}
        test_data['ct_amt_selected'] = {'text': ds.select_test.ct_text_amt.values,
                                        'label': ds.select_test.ct_label.values}

    return train_data, test_data


def fit_classifier(train_text, train_label, test_text, test_label, report=True, train='comb'):
    """
    Given training data and test data
    """
    if (len(train_text) == 0 or len(test_text) == 0):  # not generating any counterfactual examples
        return 0.0

    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    if (train == 'comb'):
        X = vec.fit_transform(list(train_text) + list(test_text))
        X_train = vec.transform(train_text)
        X_test = vec.transform(test_text)
    elif (train == 'train'):
        X_train = vec.fit_transform(list(train_text))
        X_test = vec.transform(test_text)

    clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, train_label)

    if (report):
        print(classification_report(test_label, clf.predict(X_test)))
        return clf, vec
    else:
        result = classification_report(test_label, clf.predict(X_test), output_dict=True)
        return float('%.3f' % result['accuracy'])


def classification_performance(train_data, test_data):
    """
    train: original, ct_auto, original+ct_auto
    test: original, ct_auto, ct_amt
    ct_text_auto: if no causal words or no antonym substitutions, then not use it;
    Not every text could generate a counterfactual text;
    """

    df_result = pd.DataFrame({'Sample_size': [0] * 6,
                              'Original': [0] * 6,
                              'Counterfactual': [0] * 6})

    # for flag, col in zip(['predicted_causal', 'annotated_causal', 'all_causal',"cate_causal"],
    #                      ['identified_causal', 'causal', 'all_causal', "cate"]):

    df_result.rename(index={i: f for i, f in enumerate(
        ['original', 'original+ctf_by_predicted_causal', 'original+ctf_by_annotated_causal',
         'original+ctf_by_all_causal', 'original+ctf_by_human', "original+ctf_by_cate_causal"])}, inplace=True)

    for train_flag, test_flag in product(
            ['original', 'original+ctf_by_predicted_causal', 'original+ctf_by_annotated_causal',
             'original+ctf_by_all_causal', 'original+ctf_by_human', "original+ctf_by_cate_causal"], ['Original', 'Counterfactual']):
        try:
            df_result.loc[train_flag, 'Sample_size'] = len(train_data[train_flag]['label'])
            df_result.loc[train_flag, test_flag] = fit_classifier(train_data[train_flag]['text'],
                                                                  train_data[train_flag]['label'],
                                                                  test_data[test_flag]['text'],
                                                                  test_data[test_flag]['label'], report=False)
        except:  # no human annotated counterfactual training data for kindle
            df_result.loc[train_flag, 'Sample_size'] = np.NaN
            df_result.loc[train_flag, test_flag] = np.NaN

    return df_result

def here():
    ...
if __name__=='__main__':
    data_name = "IMDB-S"
    # # data_name = "IMDB-L"
    # # data_name="KINDLE"
    #
    print(data_name, "save_cate")
    save_cate(data_name,scale=False)
    print(data_name, "get_cate_wd")
    get_lexicon_wd(data_name)
    # get_top_wd(data_name)
    # # get_cate_wd(data_name)
    # get_cate_wd_same_coef(data_name)
    print(data_name, "generate_ct_sentences")
    generate_ct_sentences(data_name)
    # # ds = load_pkl("data_before11.3/IMDB-S/ori/ds_imdb_sent.pkl")
    ds = load_pkl(Config.DATA_DIC[data_name])
    train_data, test_data = organize_data(ds, limit='')
    df_result = classification_performance(train_data, test_data)
    print(df_result)


    # data_name="IMDB-S"
def sample_wds(i):
    ds = load_pkl(Config.DATA_DIC[data_name])
    for text,cate,cate_wd,ct_text_causal,ct_causal_wds in zip(ds.train["text"].values.tolist()[i:],
                            ds.train["ct_text_cate"].values.tolist()[i:],
                          ds.train["cate_wds"].values.tolist()[i:],
                        ds.train["ct_text_causal"].values.tolist()[i:],
                        ds.train["ct_causal_wds"].values.tolist()[i:],
                          ):
        print("original text: ",text)
        print("cate:", cate,"======= generated by",cate_wd)
        print("annotated_causal:", ct_text_causal, "======= generated by", ct_causal_wds)
        break







    # s=0
    # y=0
    # for i,(cate,causal,iden,text) in enumerate(zip(ds.train.cate_wds.values.tolist()[:50],
    #                                                ds.train.causal_wds.values.tolist()[:50],
    #                                                ds.train.identified_causal_wds.values.tolist()[:50],
    #                                                ds.train.text.values.tolist())[:50]):
    #     # if len(cate)!=0:
    #     #     cate = sum([list(x.keys()) for x in cate],[])
    #
    #     if cate==causal:
    #         s+=1
    #     else:
    #         print(i, text, "\n", cate, causal)
    #
    #
    #     if iden==causal:
    #         y+=1


# def sample(i):
#     stop_lst = stopwords.words('english')
#     for ids, labels, ts, ys, index_vocab in zip(dataset_id[i:],
#                                                    ds.cate[i:],
#                                                    delta_T[i:],
#                                                    delta_Y[i:],
#                                                    index_vocab_sens[i:]):
#         ts = [[x] for x in ts]
#         ys = [[x] for x in ys]
#         ts = min_max_scaler.fit_transform(np.asarray(ts))
#         ys = min_max_scaler.fit_transform(np.asarray(ys))
#         ts = np.squeeze(ts,axis=-1)
#         ys = np.squeeze(ys,axis=-1)
#         old_mask_tokens = tokenizer.convert_ids_to_tokens([ids[position] for position in index_vocab])
#         print(old_mask_tokens)
#         unstop_tokens_postion = []
#         for k, mask_token in enumerate(old_mask_tokens):
#             if mask_token in stop_lst:
#                 pass
#             else:
#                 unstop_tokens_postion.append(k)
#         print(unstop_tokens_postion)
#         labels = np.array(labels)
#         labels = labels[unstop_tokens_postion]
#         ts = np.array(ts)
#         ts = ts[unstop_tokens_postion]
#         ys = np.array(ys)
#         ys = ys[unstop_tokens_postion]
#         index_vocab = np.array(index_vocab)
#         index_vocab = index_vocab[unstop_tokens_postion]
#
#
#         imp_word = tokenizer.convert_ids_to_tokens(ids[index_vocab[np.argmax(labels)]])
#         # print(np)
#         tokens = tokenizer.convert_ids_to_tokens(ids)
#         print(tokens)
#         # print("cft text:", ds.train_ct.text.values.tolist()[i])
#         print("the most important word:", imp_word, np.max(labels), ts[np.argmax(labels)])
#         print("the fanyici of most important word:", ds.antonym_vocab[ds.antonym_vocab["term"]==imp_word]["antonyms"])
#
#         print("causal_words:", ds.train.causal_wds.values.tolist()[i])
#         print("identified causal words", ds.train.ct_identified_causal_wds.values.tolist()[i])
#         print("word, deltaY/deltaT, deltaY, deltaT")
#
#         for index, lbl, wei, y in zip(index_vocab, labels, ts, ys):
#             token = tokenizer.convert_ids_to_tokens(ids[index])
#             print(token,lbl,y,wei)
#         break