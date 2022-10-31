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


def load_pkl(path):
    pickle_file = open(path,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data


def save_cate(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    delta_T = ds.deltaT_sens
    delta_Y = ds.deltaY_sens
    att_weight = delta_T
    att_label = []
    for ts,ys in zip(att_weight,delta_Y):
        labels = []
        for t,y in zip(ts,ys):
            try:
                labels.append(y/t)
            except:
                print(y,t)
        att_label.append(labels)
    ds.cate = att_label
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


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

# def word2vec(data_name):
#     data_name = "IMDB-S"
#     ds = load_pkl(Config.DATA_DIC[data_name])
#     train_text = ds.train.text.values.tolist()
#     train_text_split = [utils.simple_preprocess(text) for text in train_text]
#     train_model = Word2Vec(train_text_split, window=5, min_count=5, workers=4)
#     print(train_model.wv.most_similar(['good'],topn=3))
#     # [('bad', 0.9959428906440735), ('funny', 0.9944429397583008), ('actually', 0.9895380139350891)]


def get_cate_wd(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    vocab_lst = ds.antonym_vocab.term.values.tolist()
    df = ds.train
    sentences = list(df['text'].values.astype('U'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    df["dataset_id"] = dataset_id
    cate = ds.cate
    catewd_sens = []
    for ids,sen_cate in zip(dataset_id, cate):
        tokens = tokenizer.convert_ids_to_tokens(ids[1:-1])
        # top_cate = [w_cate for token,w_cate in zip(tokens,sen_cate) if token in vocab_lst]
        # median_cate = np.median(top_cate)
        catewd_sen = []

        for token,wd_cate in zip(tokens,sen_cate):

            if (token in vocab_lst):# and (wd_cate >= median_cate):
                catewd_sen.append(token)
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
    top_df = ds.top_terms
    top_lst = top_df.term.values.tolist()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ct_text_cate = []
    for cate_sen,ids in zip(ds.cate,
                            ds.train["dataset_id"].values.tolist()):
        # print(len(row["cate_wds"]))
        # break
        tokens = tokenizer.convert_ids_to_tokens(ids)
        cate_top = [w_cate for token, w_cate in zip(tokens, cate_sen) if token in top_lst]
        if len(cate_top) > 0:
            new_wds = []
            for token, wd_cate in zip(tokens, cate_sen):
                median_cate = np.median(cate_top)
                if (token in top_lst) and (wd_cate >= median_cate):
                    sub_w = list(top_df[top_df['term'] == token].antonyms.values[0].keys())
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
    # delta_T = np.load("data/AMI EVALITA 2018/deltaT_sens.npy", allow_pickle=True)
    # delta_Y = np.load("data/AMI EVALITA 2018/deltaY_sens.npy", allow_pickle=True)
    # save_cate(data_name)
    # get_cate_wd(data_name)
    # generate_ct_sentences(data_name)
    get_data_glove_syn()
    ds = load_pkl(Config.DATA_DIC[data_name])
    i= 2
    for text,cate,wd,human in zip(ds.train["text"].values.tolist()[i:],
                            ds.train["ct_text_cate"].values.tolist()[i:],
                          ds.train["cate_wds"].values.tolist()[i:],
                          ds.train_ct["text"].values.tolist()[i:]):
        print("original text: ",text)
        print("cate:", cate,"======= generated by",wd)
        print("human:",human)
        break
    # train_data, test_data = organize_data(ds, limit='')
    # df_result = classification_performance(train_data, test_data)
    # df_result




    # s=0
    # y=0
    # for i,(cate,causal,iden,text) in enumerate(zip(ds.train.cate_wds.values.tolist(),ds.train.causal_wds.values.tolist(),
    #                                                ds.train.identified_causal_wds.values.tolist(),ds.train.text.values.tolist())):
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

    # delta_T = ds.deltaT_sens
    # delta_Y = ds.deltaY_sens
    # att_weight = delta_T
    # att_label = []
    # for ts,ys in zip(att_weight,delta_Y):
    #     labels = []
    #     for t,y in zip(ts,ys):
    #         try:
    #             labels.append(y/t)
    #         except:
    #             print(y,t)
    #     att_label.append(labels)
    # ds.cate = att_label
    # pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))
    #
    # # ds = load_pkl(Config.DATA_DIC[data_name])
    # df = ds.train
    #
    # sentences = list(df['text'].values.astype('U'))
    #
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    # dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    # i = 0
    # for ids, labels, weights,ys in zip(dataset_id[i:],att_label[i:],att_weight[i:], delta_Y[i:]):
    #     imp_word = tokenizer.convert_ids_to_tokens(ids[1+np.argmax(labels)])
    #     # print(np)
    #     print("the most important word:", imp_word,np.max(labels),weights[np.argmax(labels)])
    #     print("word, deltaY/deltaT, deltaY, deltaT")
    #     for id, lbl, wei, y in zip(ids[1:-1], labels, weights, ys):
    #         token = tokenizer.convert_ids_to_tokens(id)
    #         print(token,lbl,y,wei)
    #     break