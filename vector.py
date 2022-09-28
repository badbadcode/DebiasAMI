'''
将训练集、测试集、公平测试集的向量以不同的编码形式保存下来
'''
import torch
from torch import nn
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from config import Config
import numpy as np
import os
from my_utils import *
from tqdm import tqdm
import pickle


def SaveOnehotVec():
    cleaned_fp = cleaned_fp_dic["train"]
    data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)
    x_train = data_train['cleaned_text'].values.astype('U')
    vectorizer = CountVectorizer(binary=True,min_df=7)
    vectorizer.fit(x_train)
    vectorizer_path = f"{Config.data_dir}/tokenizer/onehot_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as fw:
        pickle.dump(vectorizer, fw)

    for split in ["train","test","fair"]:
        cleaned_fp = cleaned_fp_dic[split]
        data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)
        x_train = data_train['cleaned_text'].values.astype('U')
        vec_train = vectorizer.transform(x_train)
        vec_train = torch.tensor(np.asarray(vec_train.todense()))
        vec_fp = f"{Config.VEC_DIR}/onehot/{split}.pt"
        check_dir(vec_fp)
        torch.save(vec_train,vec_fp)


def SaveTfidfVec():

    data_train = pd.read_csv(cleaned_fp_dic["train"], sep="\t", header=0)
    x_train = data_train['cleaned_text'].values.astype('U')
    vectorizer = TfidfVectorizer(
        max_features=800000,
        token_pattern=r"(?u)\b\w+\b",
        min_df=7,
        # max_df=0.1,
        analyzer='word',
        # ngram_range=(1, 5)
    )
    vectorizer.fit(x_train)
    vectorizer_path = f"{Config.data_dir}/tokenizer/tfidf_vectorizer.pkl"

    with open(vectorizer_path, 'wb') as fw:
        pickle.dump(vectorizer, fw)

    for split in ["train","test","fair"]:
        cleaned_fp = cleaned_fp_dic[split]
        data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)
        x_train = data_train['cleaned_text'].values.astype('U')
        vec_train = vectorizer.transform(x_train)
        vec_train = torch.tensor(np.asarray(vec_train.todense()))
        vec_fp = f"{Config.VEC_DIR}/tfidf/{split}.pt"
        check_dir(vec_fp)
        torch.save(vec_train,vec_fp)


def SaveBertVec(split):
    cleaned_fp = cleaned_fp_dic[split]
    data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)
    x_train = data_train['cleaned_text'].values.astype('U').tolist()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    bert = bert.to(device)

    vec_train = []

    for text in tqdm(x_train):
        encoded_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=80)
        input_ids = encoded_inputs["input_ids"]
        attention_masks = encoded_inputs["attention_mask"]
        input_ids = torch.tensor(input_ids).to(device)
        attention_masks = torch.tensor(attention_masks).to(device)

        output = bert(input_ids.unsqueeze(0),
                          token_type_ids=None,
                          attention_mask=attention_masks.unsqueeze(0), output_hidden_states=True)

        z0 = output.hidden_states[-1][:, 0, :]
        # print(z0.size()) #[1,768]
        # print(z0.squeeze(0).size())  #[768]
        vec_train.append(z0.squeeze(0).detach().cpu().numpy())

    vec_train = torch.tensor(vec_train)
    vec_fp = f"{Config.VEC_DIR}/bert/{split}.pt"
    check_dir(vec_fp)
    torch.save(vec_train,vec_fp)



cleaned_fp_dic = {"train": Config.cleaned_train_fp,
            "test": Config.cleaned_test_fp,
           "fair": Config.cleaned_test_fair_fp}



SaveOnehotVec() #3782,min_df=5:1168,6:1001,7:882
SaveTfidfVec() #3808,min_df=5:1186,6:1018
# for split in ["train","test","fair"]:
#     SaveBertVec(split) #768

# ty = "onehot"
# train = torch.load(f"{Config.VEC_DIR}/{ty}/train.pt")
# test = torch.load(f"{Config.VEC_DIR}/{ty}/test.pt")
# fair = torch.load( f"{Config.vec_dir}/{ty}/fair.pt")





