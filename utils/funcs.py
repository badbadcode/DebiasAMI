'''
[1] Shen Z ,  Cui P ,  Kuang K , et al. Causally Regularized Learning with Agnostic Data Selection Bias[J].  2017.
'''
import os
import pandas as pd
import torch
from datasets import Dataset
from utils.config import Config
import pickle
import copy
import numpy as np
import random
from utils.models import BertFT, BiGRUAttSup, BiGRUAtt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe, Vocab
from torchtext import data, datasets
from torchtext.data import get_tokenizer, Dataset, Example
from nltk.corpus import stopwords
from tqdm import trange,tqdm
from tqdm.contrib import tzip
from collections import Counter, OrderedDict


def SetupSeed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getDevice():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('using the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU , use the CPU instead.')
        device = torch.device("cpu")
    return device


def load_pkl(path):
    pickle_file = open(path,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data



def getModel(model_shortcut, data_name, seed, mode=None, att_label_type=None):
    '''
    :param model_shortcut:
    :param data_name:
    :param seed:
    :param mode: "mask","normal","mask-lexicon","weight" for b-ft
    :param att_label_type: "lexicon"/"cate"
    :return: cudaed model
    '''

    if model_shortcut == "b-ft":
        model = BertFT(data_name=data_name, seed=seed, mode=mode)
    elif model_shortcut == "gru-att":
        model = BiGRUAtt(data_name=data_name, seed=seed)
    elif model_shortcut == "gru-att-sup":
        model = BiGRUAttSup(data_name=data_name, att_label_type=att_label_type,seed=seed)

    if model.device == torch.device("cuda"):
        model.cuda()
    return model


def mask_lexicon(dataset_id, attention_mask, tokenizer):
    pos_words = open("resource/hate lexicon/positive-words.txt", encoding="ISO-8859-1").readlines()
    neg_words = open("resource/hate lexicon/negative-words.txt", encoding="ISO-8859-1").readlines()
    lexicon_lst = [x.strip() for x in pos_words + neg_words]
    lexicon_ids = tokenizer.convert_tokens_to_ids(lexicon_lst)
    new_lexicon_ids = []
    for id in lexicon_ids:
        if id != 100:
            new_lexicon_ids.append(id)
    new_sens_ids = []
    new_sens_att = []

    # print(type(dataset_id))
    for sen_ids,mask in zip(dataset_id,attention_mask):
        new_ids = []
        for id,ma in zip(sen_ids,mask):
            if ma == 1:
                if id in new_lexicon_ids:
                    new_ids.append(103)
                else:
                    new_ids.append(id)
            else:
                new_ids.append(id)

        new_sens_ids.append(new_ids)
        new_sens_att.append(mask)
    print(new_sens_ids[0])
    return new_sens_ids, new_sens_att


def mask_syn(input_ids,attention_masks,tokenizer,vocab_df):

    '''
    :param input_ids:
    :param attention_masks:
    :param tokenizer:
    :return: mask_ids_sens [num_samples, num_token_in_vocab, seq_len]
             mask_atts_sens [num_samples, num_token_in_vocab, seq_len]
             index_vocab_sens: [num_samples, num_token_in_vocab]
             prob_ids_sens:[num_samples, num_token_in_vocab, num_syns_token]
    '''

    # data_name = "IMDB-S"
    # ds = load_pkl(Config.DATA_DIC[data_name])
    # vocab_df = ds.antonym_vocab
    # df = ds.train
    # sentences = list(df['text'].values.astype('U'))
    # # print(sentences[:5])
    # df.loc[df["label"] == -1, "label"] = 0
    # labels = torch.tensor(df["label"].tolist())
    #
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    # # print("sentences", len(sentences),sentences[:5])
    # encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=model.max_len)
    # # token_vocab_lst = []
    # input_ids = encoded_inputs["input_ids"]
    # attention_masks = encoded_inputs["attention_mask"]
    mask_ids_sens = []
    mask_atts_sens = []
    index_vocab_sens = []
    prob_ids_sens = []
    print("getting the masked dataset")
    for ids, masks in zip(input_ids, attention_masks):
        tokens = tokenizer.convert_ids_to_tokens(ids)

        index_vocab = []
        new_ids_sen = []
        new_masks_sen = []
        prob_ids_sen = []

        for i, token in enumerate(tokens):
            if token in vocab_df.term.values.tolist():  # if this bert token in vocab
                index_vocab.append(i)
                syn_lst = vocab_df[vocab_df["term"] == token]["synonyms"].values.tolist()[0]
                mask_lst = [token] + syn_lst  #these tokens in this sentence (if it exsists) need to be masked

                index_prob_ids = []  # when i calculate the gold probs in this postion(i), I need sum the probs in those ids up.
                for x in mask_lst:
                    if tokenizer.convert_tokens_to_ids(x) == 100:  # [UNK]
                        continue
                    else:
                        index_prob_ids.append(tokenizer.convert_tokens_to_ids(x))

                new_ids = [103 if id in index_prob_ids else id for id in ids]   # seq_len
                # new_masks = [0 if id in index_prob_ids else masks[i] for id in ids]
                new_ids_sen.append(new_ids)
                new_masks_sen.append(masks)
                prob_ids_sen.append(index_prob_ids)

        mask_ids_sens.append(new_ids_sen)
        mask_atts_sens.append(new_masks_sen)
        index_vocab_sens.append(index_vocab)
        prob_ids_sens.append(prob_ids_sen)
    return mask_ids_sens, mask_atts_sens, index_vocab_sens,prob_ids_sens

def getTrainSenLabel(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    df = ds.train
    sentences = list(df['text'].values.astype('U'))
    # print(sentences[:5])
    if data_name=="AMI":
        labels = torch.tensor(df["misogynous"].tolist())
    else:
        df.loc[df["label"]==-1,"label"]=0
        print(df["label"].value_counts())
        labels = torch.tensor(df["label"].tolist())
    return sentences,labels

def getUnbiasedSen(data_name):
    ds = load_pkl(Config.DATA_DIC[data_name])
    if data_name == "AMI":
        df = ds.unbiased
        sentences = list(ds.unbiased['text'].values.astype('U'))
    elif data_name == "IMDB-S":
        df = ds.test_ct
        sentences = list(ds.test_ct['text'].values.astype('U'))
    else:
        df = ds.test
        sentences = list(ds.test['ct_text_amt'].values.astype('U'))
    return sentences, df


def getVocab(data_name):
    # data_name = "IMDB-S"
    ds = load_pkl(Config.DATA_DIC[data_name])
    df = ds.train
    sentences = list(df['text'].values.astype('U'))
    tokenizer = get_tokenizer("basic_english")

    counter = Counter(sum([tokenizer(sen) for sen in sentences], []))
    # sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # ordered_dict = OrderedDict(sorted_by_freq_tuples)
    data_vocab = Vocab(counter, min_freq=5, vectors=GloVe(name='6B', dim=300))

    vectors = data_vocab.vectors

    ds.train_embed_matrix = vectors
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))

    return data_vocab

def getGlovePaddedIdsMasks(data_name, sentences, model, split):
    ds = load_pkl(Config.DATA_DIC[data_name])
    tokenizer = get_tokenizer("basic_english")
    pos_words = open("resource/hate lexicon/positive-words.txt", encoding="ISO-8859-1").readlines()
    neg_words = open("resource/hate lexicon/negative-words.txt", encoding="ISO-8859-1").readlines()
    pos_words = [x.strip() for x in pos_words]
    neg_words = [x.strip() for x in neg_words]

    cate_dic = ds.lg_cate
    data_vocab = getVocab(data_name)

    input_ids = []
    attention_masks = []
    max_l = []
    for i,sen in enumerate(sentences):
        words = tokenizer(sen)
        
        max_l.append(len(words))
        
        if len(words) > model.max_len:
            words = words[:model.max_len]
        ids = []
        atts = []
        for word in words:
            id = data_vocab.stoi[word]
            ids.append(id)
            if split in ["test","unbiased"]:
                atts.append((0,0))
            else:
                if model.att_label_type =="cate":
                    if word in cate_dic.keys():
                        # print(cate_dic[word])
                        t,y = cate_dic[word][i]
                        # print(t,y)
                        atts.append((t,y/t))
                    else:
                        atts.append((0,0))
                elif model.att_label_type == "lexicon":
                    if (word in pos_words) and id != 0:
                        atts.append((1, 1))
                    elif (word in neg_words) and id != 0:
                        atts.append((1, 1))
                    else:
                        atts.append((0, 0))
                else:
                    atts.append((0, 0))

        if len(ids)<model.max_len:
            ids = ids + [1]*(model.max_len-len(ids))
            atts = atts + [(0,0)]*(model.max_len-len(atts))
        input_ids.append(ids)
        attention_masks.append(atts)
        # print(words,"\n",ids,"\n",atts)
        # break
    print("How many is the len of sentence longer than 128:", len(np.where(np.asarray(max_l)>128)[0].tolist())) #6
    return input_ids,attention_masks

def getGloveTrainDevSplitTensor(data_name, model):

    sentences,labels = getTrainSenLabel(data_name)
    # print(len(input_ids), len(input_ids[0]), len(input_ids[1]))  # 8173 80 80
    # print(len(attention_masks), len(attention_masks[0]), len(attention_masks[1]))  # 8173 80 80

    input_ids, attention_weight = getGlovePaddedIdsMasks(data_name, sentences, model, split="train")
    X = [[id,att] for id,att in zip(input_ids,attention_weight)]
    # print("X", X[0])
    X_train, X_dev, y_train, y_dev = train_test_split(X, labels, test_size=0.2)
    # print("X_train", X_train[0])

    X_train_ids = [x[0] for x in X_train]
    X_train_att = [x[1] for x in X_train]
    # print("X_train_ids", X_train_ids[0]) #torch.Size([6538, 80])
    # print("X_train_att", X_train_att[0]) #torch.Size([6538, 80, 2])
    X_dev_ids = [x[0] for x in X_dev]
    X_dev_att = [x[1] for x in X_dev]
    # print(torch.tensor(X_train_ids).size())
    # print(torch.tensor(X_train_att).size())
    train_tensor_dataset = TensorDataset(torch.tensor(X_train_ids),
                                         torch.tensor(X_train_att),
                                         torch.tensor(y_train))
    dev_tensor_dataset = TensorDataset(torch.tensor(X_dev_ids),
                                        torch.tensor(X_dev_att),
                                       torch.tensor(y_dev))

    return train_tensor_dataset, dev_tensor_dataset


def getBertMaskedTensor(sentences, labels, model,vocab_df):
    tokenizer = AutoTokenizer.from_pretrained(model.model_name, do_lower_case=True)
    # print("sentences", len(sentences),sentences[:5])
    encoded_inputs = tokenizer(sentences, padding='max_length',truncation=True, max_length=model.max_len)
    input_ids = encoded_inputs["input_ids"]
    attention_masks = encoded_inputs["attention_mask"]
    
    new_input_ids,new_attention_masks,_,_ = mask_syn(input_ids,attention_masks, tokenizer,vocab_df) #[num_samples, seq_len-2, seq_len]
    new_input_ids = torch.tensor(sum(new_input_ids,[]))
    new_attention_masks = torch.tensor(sum(new_attention_masks, []))
    
    new_labels = [[label]*len(new_mask_lst) for new_mask_lst,label in zip(new_attention_masks,labels)]
    new_labels = torch.tensor(sum(new_labels, []))
    
    tensor_dataset = TensorDataset(new_input_ids, new_attention_masks, new_labels)
    
    return tensor_dataset

def getBertTrainDevSplitTensor(data_name, model):

    sentences, labels = getTrainSenLabel(data_name)

    tokenizer = AutoTokenizer.from_pretrained(model.model_name, do_lower_case=True)
    # print("sentences", len(sentences),sentences[:5])
    encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=model.max_len)
    input_ids = encoded_inputs["input_ids"]
    attention_masks = encoded_inputs["attention_mask"]

    if model.mode == "weight":
        ds = load_pkl(Config.DATA_DIC[data_name])
        weight = ds.train["weight"].values.tolist()
        weight = [[w] for w in weight]
        X = [[id, att, w*len(id)] for id, att, w in zip(input_ids, attention_masks, weight)]
        X_train, X_dev, y_train, y_dev = train_test_split(X, labels, test_size=0.2)
        X_train_ids = [x[0] for x in X_train]
        X_train_att = [x[1] for x in X_train]
        X_train_w = [x[2] for x in X_train]
        # print("X_train_ids", X_train_ids[0]) #torch.Size([6538, 80])
        # print("X_train_att", X_train_att[0]) #torch.Size([6538, 80, 2])
        X_dev_ids = [x[0] for x in X_dev]
        X_dev_att = [x[1] for x in X_dev]
        X_dev_w = [x[2] for x in X_dev]
        train_tensor_dataset = TensorDataset(torch.tensor(X_train_ids),
                                             torch.tensor(X_train_att),
                                             torch.tensor(y_train),
                                             torch.tensor(X_train_w)
                                             )
        dev_tensor_dataset = TensorDataset(torch.tensor(X_dev_ids), 
                                           torch.tensor(X_dev_att),
                                           torch.tensor(y_dev), 
                                           torch.tensor(X_dev_w))
    else:
        if model.mode == "mask-lexicon":
            input_ids, attention_masks = mask_lexicon(input_ids, attention_masks, tokenizer)
        X = [[id, att] for id, att in zip(input_ids, attention_masks)]
        X_train, X_dev, y_train, y_dev = train_test_split(X, labels, test_size=0.2)
        X_train_ids = [x[0] for x in X_train]
        X_train_att = [x[1] for x in X_train]
        X_dev_ids = [x[0] for x in X_dev]
        X_dev_att = [x[1] for x in X_dev]

        train_tensor_dataset = TensorDataset(torch.tensor(X_train_ids),
                                             torch.tensor(X_train_att),
                                             torch.tensor(y_train))
        dev_tensor_dataset = TensorDataset(torch.tensor(X_dev_ids),
                                        torch.tensor(X_dev_att),
                                           torch.tensor(y_dev))

    return train_tensor_dataset, dev_tensor_dataset


def getTrainDevLoader(model,data_name):
    if model.model_shortcut in ["b-ft"]:
        if model.mode in ["normal", "mask-lexicon", "weight"]:
            print("Loading the train data...")
            train_data,dev_data = getBertTrainDevSplitTensor(data_name, model)
        elif model.mode == "mask":  # mask all words and syns
            train_data = getBertMaskedTensor(Sen_train, Lbl_train, model, ds.antonym_vocab)
            dev_data = getBertMaskedTensor(Sen_dev, Lbl_dev, model, ds.antonym_vocab)
    elif model.model_shortcut in ["gru-att", "gru-att-sup"]:
        train_data, dev_data = getGloveTrainDevSplitTensor(data_name, model)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=model.batch_size)
    dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=model.batch_size)

    return train_dataloader, dev_dataloader



def getTestLoader(model, data_name, test_name):

    ds = load_pkl(Config.DATA_DIC[data_name])
    if test_name =="test":
        df = ds.test
        sentences = list(df['text'].values.astype('U'))
    else:
        sentences, df = getUnbiasedSen(data_name)

    if model.model_shortcut in ["b-ft"]:
        tokenizer = AutoTokenizer.from_pretrained(model.model_name, do_lower_case=True)
        encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=model.max_len)
        input_ids = encoded_inputs["input_ids"]
        attention_masks = encoded_inputs["attention_mask"]
        # X = [[id, att] for id, att in zip(input_ids, attention_masks)]

    elif model.model_shortcut in ["gru-att-sup", "gru-att"]:
        input_ids,attention_masks = getGlovePaddedIdsMasks(data_name, sentences, model, split=test_name)
        # X = [[id, att] for id, att in zip(input_ids, attention_masks)]
    else:
        print("this model_shortcut has not been defined.")

    if "label" in df.columns:
        df.loc[df["label"] == -1, "label"] = 0
        print(df["label"].value_counts())
        labels = torch.tensor(df["label"].tolist())
        test_data = TensorDataset(torch.tensor(input_ids),
                                  torch.tensor(attention_masks),
                                  labels)
    elif "misogynous" in df.columns:
        labels = torch.tensor(df["misogynous"].tolist())
        test_data = TensorDataset(torch.tensor(input_ids),
                                  torch.tensor(attention_masks),
                                  labels)
    else:
        test_data = TensorDataset(torch.tensor(input_ids),
                                  torch.tensor(attention_masks))

    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=model.batch_size)

    return test_dataloader


def save_weight2ds(mask_model, data_name):
    print("sequential train data loader")
    ds = load_pkl(Config.DATA_DIC[data_name])
    sentences, labels = getTrainSenLabel(data_name)
    tokenizer = AutoTokenizer.from_pretrained(mask_model.model_name, do_lower_case=True)
    # print("sentences", len(sentences),sentences[:5])
    encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=mask_model.max_len)
    input_ids = encoded_inputs["input_ids"]
    attention_masks = encoded_inputs["attention_mask"]

    input_ids, attention_masks = mask_lexicon(input_ids, attention_masks, tokenizer)
    tensor_dataset = TensorDataset(torch.tensor(input_ids),
                                   torch.tensor(attention_masks),
                                   torch.tensor(labels))
    train_mask_dataloader = DataLoader(tensor_dataset,
                                       sampler=SequentialSampler(tensor_dataset),
                                       batch_size=mask_model.batch_size)
    mask_model.eval()  # prep model for evaluation
    dev_logits = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(train_mask_dataloader, ncols=70)):
            b_input_ids = batch[0].to(mask_model.device)
            b_input_mask = batch[1].to(mask_model.device)
            dev_loss_dic = mask_model(input_ids=b_input_ids, input_mask=b_input_mask)
            dev_logits.append(dev_loss_dic["logits"].cpu().numpy())  # [tensor(batch_size,NUM_LABELS),tensor(batch_size,NUM_LABELS),.....]
    # calculate average loss over an epoch (all batches)
    dev_logits = [x.tolist() for x in dev_logits]  # [num_batch,batch_size,num_labels]
    dev_logits_flat = sum(dev_logits, [])
    one_hot_weights = 1 / np.asarray(dev_logits_flat)
    # one_hot_label = torch.nn.functional.one_hot(b_labels,num_classes=2)
    weight = np.where(labels == 1,
                      one_hot_weights[:, 1],
                      one_hot_weights[:, 0])
    print(type(weight))
    print(weight)
    ds.train["weight"] = weight
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def check_dir(vec_fp):
    dir = "/".join(vec_fp.split("/")[:-1])
    if not os.path.exists(dir):
        print("prepare makingdirs", dir)
        os.makedirs(dir)
    else:
        pass


def cal_Jbs(X, w, I,fem_index):
    '''
    计算论文中的Jb，返回包含一个序列，是j=1 -> j=p的 的Jb

    :param X: 输入 n X p
    :param w: 小w n X 1
    :param I: 处理矩阵，表示是否处理，值只有 0， 1 n X p
    :return: Jbs 所有Jb，维度为[p, p] 不是Jb_j
    '''
    W = w * w
    Jbs = []

    for j in range(X.shape[1]):
        if fem_index[j]==1:
            X_minus_j = X.clone().detach()
            X_minus_j[:, j] = 0
            Ij = I[:, j] #(n)
            Jb_item = (X_minus_j.T @ (W * Ij) / (W.T @ Ij) - (X_minus_j.T @ (W * (1 - Ij))) / (W.T @ (1 - Ij)))
            # print(Jb_item.size()) #torch.Size([882])
            Jbs.append(Jb_item.unsqueeze(0)) # Jb_item.unsqueeze(0): torch.Size([1,882])
        else:
            continue
    return torch.cat(Jbs, dim=0) #(p,p)


def cal_loss_of_confounder_from_Jbs(Jbs):
    '''
    将Jbs求二范数再平方再求和

    :param Jbs:
    :return: 计算结果
    '''

    temp = Jbs.norm(2, 1) ** 2 # 对行向量求二范数 （p,p）==>(p)
    temp = temp.sum()

    return temp


def cal_partial_Jbs_partial_W(X, w, I, device, fem_index):
    '''
    计算论文中的 ∂J(b)/∂ω

    :param X: 输入数据
    :param w: 小w
    :param I: 处理矩阵
    :param fem_index:性别
    :return: ∂J(b)/∂ω 列表，包含p个，维度为  [p, p, n]
    '''
    pJpWs = []
    # print("w",w.size()) #(n)
    W = w * w
    the_one = torch.ones(X.shape[1], 1).to(device)  # (p,1)
    for j in range(X.shape[1]): # X的特征维度，依次将每个变量作为处置变量 循环内部的是论文里的公式
        if fem_index[j] == 1:
            X_minus_j = X.clone().detach().to(device)
            X_minus_j[:, j] = 0
            Ij = I[:, j].view(1, -1)  # 当第j个变量作为处置变量时，所有样本的处置情况（1/0） #（1，n) 这和我推导时不一样
            pJpw_item1 = (X_minus_j.T * (the_one @ Ij)) * (W @ Ij.T) - (X_minus_j.T @ (W * Ij).T @ Ij) #w * w * Ij size:(1,n)
            pJpw_item1 /= (W @ Ij.T) ** 2  #(p,n)

            Ij = (1 - Ij)  #(1,n)
            pJpw_item2 = (X_minus_j.T * (the_one @ Ij)) * (W @ Ij.T) - (X_minus_j.T @ (W * Ij).T @ Ij)
            pJpw_item2 /= (W @ Ij.T) ** 2 #(p,n)

            pJpw_item = pJpw_item1 - pJpw_item2  # (p,n)
            pJpWs.append(pJpw_item)
        else:
            continue

    return torch.stack(pJpWs, dim=0) #(p,p,n)


def cal_partial_J_parital_w(X, Y, w, I, pred, fem_index, lambda1=1, lambda2=1, lambda5=1, device='cuda'):
    '''
    计算∂J(ω)/∂ω

    :param X: 输入
    :param Y: 目标输出
    :param w: 小w
    :param I: 处理矩阵
    :param pred: 这个就是 Xβ
    :return: 对于w的梯度，维度为 n
    '''

    # 计算第一部分
    # 这里是不是可以用神经网络计算出来的loss HMTQ：0422 15：50
    # log_expYX = (1 - 2 * Y) * pred
    # log_expYX = torch.log(1 + torch.exp(log_expYX))
    # part1 = w * w * log_expYX
    # 2022 0425 9:49 改成交叉熵

    # print("X.size", X.size()) #torch.Size([3200, 882])

    part1 = (Y) * torch.log(pred[:, 1]) + (1 - Y) * torch.log(pred[:, 0]) # (n,1)
    part1 = -part1  # 交叉熵是负的
    part1 = 2 * w * part1
    # print("w",w.size()) #torch.Size([3200])
    # print("(part1.size()",part1.size()) # torch.Size([3200])
    # part1 = part1.sum()

    # 计算第二部分
    the_one = torch.ones(X.shape[1], 1).to(device) #(p,1)
    pJbs_pW = cal_partial_Jbs_partial_W(X, w, I, device, fem_index) #(p,p,n) #HMY：我认为论文这里写错了，其实这里计算的是对W的梯度_
    Jbs = cal_Jbs(X, w, I,fem_index) #(p,p)
    part2 = torch.bmm((pJbs_pW * (the_one @ w.view(1, -1))).permute(0, 2, 1), Jbs.view(Jbs.shape[0], -1, 1)).squeeze(-1)
    # w.view(1, -1) #w变成行向量,（1, n）
    # (the_one @ w.view(1, -1)) # (p,n)
    # (pJbs_pw * (the_one @ w.view(1, -1))).permute(0, 2, 1) #(f_num,n,p)
    # Jbs.view(Jbs.shape[0], -1, 1)  #(f_num,p,1)
    # (f_num,n,p)(f_num,p,1) = (p,n,1)==>(p.n)

    # print("∂J(ω)/∂ω: part2",part2.size()) # torch.Size([882, 3200])

    part2 = part2.sum(0)  # 这里sum之前要增加一个加权/选择循环的时候略过fem_index为0的地方2022-9-26 00:43:58

    part2 *= 4*lambda1

    # 计算第三部分
    part3 = 4 * lambda2 * w * w * w

    # 计算第四部分
    delta = (w*w).sum() - 1
    part4 = 4 * lambda5 * delta * w


    # print("part1",part1.size()) #(n)
    # print("part2",part2.size())
    # print("part3",part3.size())
    # print("part4",part4.size())

    # print('1', part1[:2])
    # print('2', part2[:2])
    # print('3', part3[:2])
    # print('4', part4)
    return part1 + part2 + part3 + part4


def cal_Jw(X, Y, w, I, pred, fem_index, lambda1=1, lambda2=1, lambda5=1, device='cuda'):
    '''
    计算J(w)

    :param X: 输入
    :param Y: 期望输出
    :param w: 小w
    :param I: 处理矩阵
    :param pred: 预测值 这个就是 Xβ
    :param fem_index: 性别index
    :param lambda1:
    :param lambda2:
    :param lambda5:
    :return: 损失， 1 维
    '''
    W = w * w

    # # 2022 0425 9:49 改成交叉熵
    # log_expYX = (1 - 2 * Y) * pred
    # log_expYX = torch.log(1 + torch.exp(log_expYX)).to(device)
    # part1 = W * log_expYX
    # part1 = part1.sum()

    # print("X", X.size(),   # torch.Size([3200, 882])
    #       "Y", Y.size(),  #  torch.Size([3200])
    #       "pred", pred.size() # pred torch.Size([3200, 2])
    # )
    part1 = (Y) * torch.log(pred[:, 1]) + (1 - Y) * torch.log(pred[:, 0])

    part1 = -part1
    part1 = w * w * part1

    # print("J(w) part1.size()", part1.size())  # torch.Size([3200])
    part1 = part1.sum()

    # print("J(w) part1.size()", part1.size())  # tensor(2139.2676, device='cuda:0'), torch.Size([])

    '''
    这里要增加：乘以0-1向量（before sum）
    '''
    # print("J(w) cal_Jbs(X, w, I)", cal_Jbs(X, w, I).size()) #torch.Size([882, 882])

    part2 = cal_loss_of_confounder_from_Jbs(cal_Jbs(X, w, I,fem_index))

    # print("J(w) part2.size()",part2.size())  # torch.Size([])

    part3 = (W.norm(2) ** 2)

    part4 = ((W.sum()-1) ** 2)
    # print("lambda5", lambda5)
    # print("original  part4 of loss2", part4)

    result = part1 + lambda1 * part2 + lambda2 * part3 + lambda5 * part4
    part2 = lambda1 * part2
    part3 = lambda2 * part3
    part4 = lambda5 * part4
    return result, [part1, part2, part3, part4]


def update_w(w, w_g, lr, clip):
    '''
    更新小w

    :param w: 小w
    :param w_g: 小w的梯度
    :param lr: 学习率
    :return: 新w
    '''
    w_g = w_g.clamp(-clip, clip)
    w = w - lr * w_g
    return w


def update_w_one_step(X, Y, w, I, pred, lambda1, lambda2, lambda5, lr, fem_index, device, clip=50):
    '''
    更新一步，即更新一次w

    :param X: 输入
    :param Y: 期望输出
    :param w: 小w
    :param I: 处理矩阵
    :param pred: 预测值
    :param lambda1: confounder值的超参
    :param lambda2: w的正则化超参
    :param lambda5: 为了让w不为0的超参
    :param lr:
    :param fem_index:作为处置变量的单词位置为1
    :return: new_w 新w, loss 当前步的损失
    '''
    fem_index = torch.tensor(fem_index).to(device)
    w_g = cal_partial_J_parital_w(X, Y, w, I, pred, fem_index, lambda1=lambda1, lambda2=lambda2, lambda5=lambda5, device=device)
    new_w = update_w(w, w_g, lr, clip)
    loss, loss_detail = cal_Jw(X, Y, new_w, I, pred,fem_index, lambda1=lambda1, lambda2=lambda2, lambda5=lambda5, device=device)
    return new_w, loss, loss_detail

def get_sentence_vecs(vec_type, data_name):
    '''
    获得 aclImdb数据级经过Bert编码后的句向量

    :param vec_type: 向量类型 包括['last_hidden_state_first', 'pooler_output']
    :return: train_dataset, test_dataset
    '''
    ds = load_pkl(Config.DATA_DIC[data_name])
    # path = "data/AMI EVALITA 2018/vector" if path is None else path

    train_X = torch.load(f"{Config.DATA_DIR[data_name]}/vector/{vec_type}/train.pt")
    # train_X = torch.load(os.path.join(path, vec_type, 'train.pt'))
    data_train = ds.train
    train_label = data_train['label'].values.astype('int').tolist()
    train_label = [0 if l==-1 else l for l in train_label]
    train_Y = torch.LongTensor(train_label)

    test_X = torch.load(f"{Config.DATA_DIR[data_name]}/vector/{vec_type}/test.pt")
    data_test = ds.test
    test_label = data_test['label'].values.astype('int').tolist()
    test_label = [0 if l == -1 else l for l in test_label]
    test_Y = torch.LongTensor(test_label)

    test_fair_X = torch.load(f"{Config.DATA_DIR[data_name]}/vector/{vec_type}/unbiased.pt")
    if data_name == "AMI":
        unbiased_label = list(ds.unbiased['label'].values.astype('int'))
    elif data_name == "IMDB-S":
        unbiased_label = list(ds.test_ct['label'].values.astype('int'))
    else:
        unbiased_label = list(ds.test['ct_label'].values.astype('int'))

    # data_test_fair = pd.read_csv(Config.cleaned_test_fair_fp, sep="\t", header=0)
    unbiased_label = [0 if l == -1 else l for l in unbiased_label]
    test_fair_Y = torch.LongTensor(unbiased_label)


    train_dataset = Dataset.from_dict({
        'x': train_X,
        'labels': train_Y
    })

    test_dataset = Dataset.from_dict({
        'x': test_X,
        'labels': test_Y
    })

    test_fair_dataset = Dataset.from_dict({
        'x': test_fair_X,
        'labels': test_fair_Y
    })

    return train_dataset, test_dataset, test_fair_dataset
# def get_sentence_vecs(vec_type, path=None):
#     '''
#     获得 aclImdb数据级经过Bert编码后的句向量
#
#     :param vec_type: 向量类型 包括['last_hidden_state_first', 'pooler_output']
#     :return: train_dataset, test_dataset
#     '''
#
#     path = "data/AMI EVALITA 2018/vector" if path is None else path
#     train_X = torch.load(os.path.join(path, vec_type, 'train.pt'))
#     data_train = pd.read_csv(Config.cleaned_train_fp, sep="\t", header=0)
#     train_Y = torch.LongTensor(data_train['misogynous'].values.astype('int').tolist())
#
#     test_X = torch.load(os.path.join(path, vec_type, f'test.pt'))
#     data_test = pd.read_csv(Config.cleaned_test_fp, sep="\t", header=0)
#     test_Y = torch.LongTensor(data_test['misogynous'].values.astype('int').tolist())
#
#     test_fair_X = torch.load(os.path.join(path, vec_type, f'fair.pt'))
#     data_test_fair = pd.read_csv(Config.cleaned_test_fair_fp, sep="\t", header=0)
#     test_fair_Y = torch.LongTensor(data_test_fair['misogynous'].values.astype('int').tolist())
#
#     train_dataset = Dataset.from_dict({
#         'x': train_X,
#         'labels': train_Y
#     })
#
#     test_dataset = Dataset.from_dict({
#         'x': test_X,
#         'labels': test_Y
#     })
#
#     test_fair_dataset = Dataset.from_dict({
#         'x': test_fair_X,
#         'labels': test_fair_Y
#     })
#
#     return train_dataset, test_dataset, test_fair_dataset

# def getGenderIndex(vec_type, only_female=True):
#     '''
#     :param vec_type: 编码的类型
#     :param only_female: 是否将所有单词都看做是处置变量
#     :return:
#     '''
#     if only_female:
#         baseline_swap_list = [["woman", "man"],["girls","boys"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"]]
#         female_words = [x[0] for x in baseline_swap_list]
#         if vec_type in ["onehot", "tfidf"]:
#             vectorizer_path = f"{Config.vectorizer_path}{vec_type}_vectorizer.pkl"
#             vectorizer = pickle.load(open(vectorizer_path, "rb"))
#             vocab_dic = vectorizer.vocabulary_
#             gender_index = np.zeros(len(vocab_dic.keys()))
#             for x in female_words:
#                 try:
#                     gender_index[vocab_dic[x]]=1
#                     # print(x)# no "gal","herself"
#                 except:
#                     pass
#         else:
#             gender_index = np.ones(768)
#     else: #把所有单词当做处置变量
#         if vec_type in ["onehot", "tfidf"]:
#             vectorizer_path = f"{Config.vectorizer_path}{vec_type}_vectorizer.pkl"
#             vectorizer = pickle.load(open(vectorizer_path, "rb"))
#             vocab_dic = vectorizer.vocabulary_
#             gender_index = np.ones(len(vocab_dic.keys()))
#         else:
#             gender_index = np.ones(768)
#
#     return gender_index

def getGenderIndex(vec_type, data_name, only_female=True):
    '''
    :param vec_type: 编码的类型
    :param only_female: 是否将所有单词都看做是处置变量
    :return:
    '''
    if data_name == "AMI":
        if only_female:
            baseline_swap_list = [["woman", "man"],["girls","boys"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"]]
            female_words = [x[0] for x in baseline_swap_list]
            if vec_type in ["onehot", "tfidf"]:
                vectorizer_path = f"{Config.DATA_DIR[data_name]}/tokenizer/{vec_type}_vectorizer.pkl"
                # vectorizer_path = f"{Config.vectorizer_path}{vec_type}_vectorizer.pkl"
                vectorizer = pickle.load(open(vectorizer_path, "rb"))
                vocab_dic = vectorizer.vocabulary_
                gender_index = np.zeros(len(vocab_dic.keys()))
                for x in female_words:
                    try:
                        gender_index[vocab_dic[x]]=1
                        # print(x)# no "gal","herself"
                    except:
                        pass
            else:
                gender_index = np.ones(768)
        else: #把所有单词当做处置变量
            if vec_type in ["onehot", "tfidf"]:
                vectorizer_path = f"{Config.DATA_DIR[data_name]}/tokenizer/{vec_type}_vectorizer.pkl"
                vectorizer = pickle.load(open(vectorizer_path, "rb"))
                vocab_dic = vectorizer.vocabulary_
                gender_index = np.ones(len(vocab_dic.keys()))
            else:
                gender_index = np.ones(768)
    elif data_name in ["IMDB-S", "IMDB-L", "KINDLE"]:
        pos_words = open("resource/hate lexicon/positive-words.txt", encoding="ISO-8859-1").readlines()
        neg_words = open("resource/hate lexicon/negative-words.txt", encoding="ISO-8859-1").readlines()
        lexicon_lst = pos_words + neg_words
        lexicon_lst = [x.strip() for x in lexicon_lst]
        if only_female:
            if vec_type in ["onehot", "tfidf"]:
                vectorizer_path = f"{Config.DATA_DIR[data_name]}/tokenizer/{vec_type}_vectorizer.pkl"
                # vectorizer_path = f"{Config.vectorizer_path}{vec_type}_vectorizer.pkl"
                vectorizer = pickle.load(open(vectorizer_path, "rb"))
                vocab_dic = vectorizer.vocabulary_
                lexicon_lst_vocab = []
                for x in lexicon_lst:
                    if x in list(vocab_dic.keys()):
                        lexicon_lst_vocab.append(x)

                gender_index = np.ones(len(vocab_dic.keys()))
                for x in lexicon_lst_vocab:
                    try:
                        gender_index[vocab_dic[x]]=0  # neutral words
                        # print(x)# no "gal","herself"
                    except:
                        pass
            else:
                gender_index = np.ones(768)
        else: #把所有单词当做处置变量
            if vec_type in ["onehot", "tfidf"]:
                vectorizer_path = f"{Config.DATA_DIR[data_name]}/tokenizer/{vec_type}_vectorizer.pkl"
                vectorizer = pickle.load(open(vectorizer_path, "rb"))
                vocab_dic = vectorizer.vocabulary_
                gender_index = np.ones(len(vocab_dic.keys()))
            else:
                gender_index = np.ones(768)


    return gender_index

# if __name__ == '__main__':
#     seed = 824
#     torch.manual_seed(seed)  # 为CPU设置随机种子
#     torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
#     torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
#
#     X = torch.randn(100000, 5)
#     Y = torch.zeros(X.shape[0])
#     Y[torch.rand_like(Y) > 0.5] = 1
#     pred = torch.rand(X.shape[0])
#     I = torch.zeros_like(X)
#     I[torch.rand(*X.shape) > 0.5] = 1
#     w = torch.rand(X.shape[0])
#
#     losses = []
#     lr = 1e-3
#     lr_decay = 0.5
#     lambda1 = 1e-2
#     lambda2 = 1e-3
#     lambda5 = 1e-8
#     for i in range(10):
#         w, loss, loss_detail = update_w_one_step(X, Y, w, I, pred, lambda1, lambda2, lambda5, lr, device='cpu')
#         losses.append(loss)
#         lr = lr * lr_decay
#
#     print(losses[:3], losses[-3:])
#     print((w * w)[:10])
#     print((w * w).sum())

