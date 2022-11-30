from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix
from scipy.stats import ttest_rel, binom
# from scipy.stats import chisqprob # old version
from scipy.stats import chi2
import numpy as np
from math import log
import sys
from utils.config import Config
import pickle
from tqdm import tqdm
from utils.funcs import getVocab
from nltk.corpus import stopwords



def load_pkl(path):
    pickle_file = open(path,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data


# filename = sys.argv[1]
# data_vocab = getVocab(data_name)
# words = data_vocab.itos
# stop_lst = stopwords.words('english')
#
# new_words = []
# for w in words:
#     if w not in stop_lst:
#         new_words.append(w)
def main(data_name):
    param_reg = 1.0
    param_thresh = 100000
    ds = load_pkl(Config.DATA_DIC[data_name])
    sentences = ds.train.text.values.tolist()
    if data_name == "AMI":
        label = np.asarray(ds.train.misogynous.values.tolist())
    else:
        label = np.asarray(ds.train.miso.values.tolist())
        label = (label + 1)/2
    label = label.astype(int).tolist()
    vec = CountVectorizer(min_df=5, binary=True,  max_df=.8, stop_words="english")
    vec.fit(sentences)
    vec_sens = vec.transform(sentences)
    model_y = linear_model.LogisticRegression(C=param_reg, max_iter=1000)
    model_y.fit(vec_sens, label)
    vocab_dic = vec.vocabulary_
    coef = model_y.coef_[0]  # coef.shape ==> (2865,)
    coef_dic = {}
    for k, v in vocab_dic.items():
        coef_dic[k] = coef[v]

    sorted_coef_dic = sorted(coef_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    sorted_coef_dic_400 = sorted_coef_dic[:200] + sorted_coef_dic[-200:]
    top_words_400 = [x[0] for x in sorted_coef_dic_400]
    # scores_y = model_y.predict(vec_sens)


    analyze = vec.build_analyzer()
    word_docs = {}
    vocab = {}
    vocab_count = {}
    l = 0
    for i,sen in tqdm(enumerate(sentences)):
        tokens = analyze(sen)
        for word in set(tokens):
            if word not in vocab:
                vocab[word] = len(vocab)
                vocab_count[word] = 0
            else:
                vocab_count[word]+=1
            if word not in word_docs:
                word_docs[word] = []
            word_docs[word].append(l)

        l += 1

    L = l
    V = len(vocab)
    # vocab_5 = {k: v for k, v in vocab_count.items() if v > 4}
    # V = len(vocab_5.items())

    print('%d documents, %d distinct word types' % (L, V))




    # Run matching routine for each word to calculate its p-value

    word_index_score_dic = {}
    for treatment in tqdm(list(vec.vocabulary_.keys())):
        rows = []
        cols = []
        values = []
        y = []

        # Read through input file and create a sparse matrix of word counts (excluding treatment word)
        # to train the propensity classifier
        model_t = linear_model.LogisticRegression(C=param_reg)
        for i,line in enumerate(sentences):
            tokens = analyze(line)
            # print(tokens)
            contains_treatment = 0
            for word in set(tokens):
                v = vocab[word]  # the index of the word

                if word == treatment:
                    contains_treatment = 1
                else:
                    rows.append(i)
                    cols.append(v)
                    values.append(1.)

            y.append(contains_treatment)


        data = coo_matrix((values, (rows, cols)), shape=(len(sentences), V)).toarray()
        # break
        model_t.fit(data, y)
        scores = model_t.predict_proba(data)

        # Read through the file again and create a list of propensity scores (from the classifier above)
        # and document labels (the first token of each document)

        docs_t = word_docs[treatment]

        index_dic = {}


        for index in docs_t:
            sen_t = sentences[index]

            sen_vec = vec.transform([sen_t]).toarray()
            word_index = vocab_dic[treatment]
            sen_vec[0][word_index] = 0
            pred_y = model_y.predict_proba(sen_vec)
            index_dic[index] = (scores[index][0],pred_y[0][1-label[index]])


        word_index_score_dic[treatment]=index_dic


    ds.lg_cate = word_index_score_dic
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))
    # docs.append((scores[i][1], label, contains_treatment)) #[包含t的概率，句子label, 是否包含T]

main("AMI")
# for data_name in ["IMDB-S","IMDB-L","KINDLE"]:
#     main(data_name)


# for word,index_dic in ds.lg_cate.items():
#     for index,(t,y) in index_dic.items():
#         cate = y/t
#         if cate>20:
#
#             print(word,index,t,y,cate)
#             print(sentences[index])
#
# def diff(t1,t2):
#     t1 = torch.tensor([0.2,0.1])
#     t2 = torch.tensor([0.9,0.8])
#     t22 = torch.tensor([0.5,0.4])
#     loss = -1 * t1 * torch.log(t2)  # tensor([0.0211, 0.0223])
#     loss = -1 * t1 * torch.log(t22)  # tensor([0.1386, 0.0916])
#
#     crt = torch.nn.KLDivLoss(reduction='none')
#     crt(t1.log(),t2)
#
data_name = "AMI"
ds = load_pkl(Config.DATA_DIC[data_name])
sentences = ds.train.text.values.tolist()
word_index_score_dic = ds.lg_cate


from torchtext.data import get_tokenizer, Dataset, Example
tokenizer = get_tokenizer("basic_english")
vec = CountVectorizer(min_df=5, binary=True, max_df=.8, stop_words="english")
analyze = vec.build_analyzer()

for i,sen in enumerate(sentences[1:]):
    tokens = analyze(sen)
    print(sen)
    for token in tokens:
        if token in word_index_score_dic.keys():
            t,y = word_index_score_dic[token][i]
            print(token,t,y)
    break

for k,v in word_index_score_dic.items():
    print(k)
    for sub_k, sub_v in v.items():
        print(
            sub_v[0],
              ",",
              sub_v[1],
              ",",
              sub_v[1]/sub_v[0],
              ",",
              sentences[sub_k])
    break