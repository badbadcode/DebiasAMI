from sklearn.linear_model import LogisticRegression
from utils.config import Config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os

def load_split_train():
    data_train = pd.read_csv(Config.cleaned_train_fp, sep="\t", header=0)
    data_train = data_train.sample(frac=1, random_state=42)
    x_train, x_dev, y_train, y_dev = \
        train_test_split(data_train['cleaned_text'].values.astype('U'),
                         data_train['misogynous'].values.astype('int'),
                         test_size=0.2,
                         random_state=42
                         )

    return x_train, x_dev, y_train, y_dev


def load_test():
    data_test = pd.read_csv(Config.cleaned_test_fp, sep="\t", header=0)
    data_test_fair = pd.read_csv(Config.test_fair_fp, sep="\t", names=["text", "misogynous"])
    x_test = data_test['cleaned_text'].values
    y_test = data_test['misogynous'].values
    x_test_fair = data_test_fair['text'].values
    y_test_fair = data_test_fair['misogynous'].values
    return x_test, y_test, x_test_fair, y_test_fair


def fitted_vectorizer(vec_type, x_train):
    print(f"Now we are fitting a {vec_type} vectorizer from x_train")
    if vec_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=800000,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            # max_df=0.1,
            analyzer='word',
            # ngram_range=(1, 5)
        )
        vectorizer.fit(x_train)
    elif vec_type == "onehot":
        vectorizer  = CountVectorizer(binary=True)
        vectorizer.fit(x_train)
    else:
        vectorizer = TfidfVectorizer(
            max_features=800000,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            # max_df=0.1,
            analyzer='word',
            # ngram_range=(1, 5)
        )

    return vectorizer


def get_vec_train(vectorizer, x_train, x_dev):
    vec_train = vectorizer.transform(x_train)
    vec_dev = vectorizer.transform(x_dev)
    return vec_train, vec_dev


def get_vec_test(vectorizer, x_test, x_test_fair):
    vec_test = vectorizer.transform(x_test)
    vec_test_fair = vectorizer.transform(x_test_fair)
    return vec_test, vec_test_fair

def getModel(model_name):

    if model_name == "LR":
        model = LogisticRegression(solver='sag',verbose=2)
    else:
        model = LogisticRegression(solver='sag', verbose=2)
    return model

def TopK(model,top_k,vectorizer):
    coef = model.coef_[0] # ndarray
    vocab_dic = vectorizer.vocabulary_
    coef_dic = {}
    top_k_words = []
    for k,v in vocab_dic.items():
        coef_dic[k] = coef[v]
    for k,v in sorted(coef_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top_k]:
        top_k_words.append(k)
    return top_k_words

# def TopKbert(model,k):
#
#     coef = model.coef_[0] # ndarray
#     vocab_dic = vectorizer.vocabulary_
#     coef_dic = {}
#     top_k_words = []
#     for k,v in vocab_dic.items():
#         coef_dic[k] = coef[v]
#     for k,v in sorted(coef_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top_k]:
#         top_k_words.append(k)
#     return top_k_words


def check_dir(vec_fp):
    dir = "/".join(vec_fp.split("/")[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass