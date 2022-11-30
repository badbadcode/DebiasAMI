from utils.config import Config
from utils.funcs import getTrainSenLabel, load_pkl
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch.nn as nn
import torch
from transformers import AdamW
import torch
from tqdm import trange
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

import re, numpy as np, pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from torchtext.vocab import GloVe
from sklearn import metrics


def word_cluster(sensitive_words):
    # 词向量
    glove = GloVe(name='6B', dim=300)
    new_sensitive_words = list(filter(lambda x: x in glove.itos, sensitive_words))
    sensi_vectors = np.zeros([len(new_sensitive_words), 300])
    print("get the glove vectors of sensitive words")
    for i, word in enumerate(tqdm(new_sensitive_words)):
        glove_word_index = glove.stoi[word]
        glove_word_vec = glove.vectors[glove_word_index]
        sensi_vectors[i] = glove_word_vec
    # 聚类
    n_clusters = 95
    labels = KMeans(n_clusters=n_clusters).fit(sensi_vectors).labels_
    # 输出excel
    df = pd.DataFrame([(w, labels[e]) for e, w in enumerate(new_sensitive_words)], columns=['word', 'label'])
    word_cluster = [df[df["label"] == i].word.values.tolist() for i in range(n_clusters)]

    # import numpy as np
    # from sklearn.cluster import KMeans
    # for num in range(2,155):
    #     kmeans_model = KMeans(n_clusters=num, random_state=1).fit(sensi_vectors)
    #     labels = kmeans_model.labels_
    #     score = metrics.davies_bouldin_score(sensi_vectors, labels)
    #     print(num, score)
    '''
    2 20.50710927978315
    3 16.503311337045904
    4 13.666629263929881
    5 12.223453520212185
    6 10.808016942041885
    '''

    return word_cluster


class WeightLinear(nn.Module):
    def __init__(self, sample_num):
        super(WeightLinear, self).__init__()
        # self.w = nn.Parameter(1/1/torch.ones(gender0_num,), requires_grad=True)
        self.sample_num = sample_num
        self.w_all = nn.Parameter(torch.zeros(sample_num, ), requires_grad=True)
        self.beta = 1
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, XC, XT, gender0_row, gender1_row):
        # print("XC", XC.size())  # torch.Size([2777, 78])
        # print("xt_avg", xt_avg.size())  # torch.Size([78])
        # print("W",w.size())
        with torch.no_grad():
            gender1_num = gender1_row.size()[0]
            w = torch.exp(self.w_all.unsqueeze(-1))
            w_t = w[gender1_row]
            xt_avg = torch.matmul(XT.t(), w_t) / gender1_num
            print(xt_avg[:5])

        gender0_num = gender0_row.size()[0]
        new_w = torch.exp(self.w_all.unsqueeze(-1))
        w_c = new_w[gender0_row]
        xc_avg = torch.matmul(XC.t(), w_c) / gender0_num  # (cfder_num, gender0_num) (gender0_num, 1)
        print(xc_avg[:5])

        # ones = torch.ones_like(w.t())
        #
        # weight_sum = torch.matmul(ones,w)
        # weight_reg_loss = torch.pow(weight_sum.squeeze() - 1, 2)

        weight_loss = torch.nn.functional.mse_loss(input=xc_avg.squeeze(-1),
                                                   target=xt_avg.squeeze(-1),
                                                   reduction='sum')
        # loss = weight_loss + self.beta*weight_reg_loss
        loss = weight_loss
        return {"loss": loss,
                # "weight_loss":weight_loss,
                # "weight_reg_loss":weight_reg_loss
                }


def judgeCon1(vec_word_gender0, label_gender0):
    res_cond1 = False
    sum_res = vec_word_gender0 + label_gender0
    count = Counter(sum_res.tolist())
    word0_y0_num = count[-1]
    word0_y1_num = count[1]
    word1_y0_num = count[0]
    word1_y1_num = count[2]
    word_label = np.array([[word0_y0_num, word0_y1_num],
                           [word1_y0_num, word1_y1_num]])

    p_value = stats.chi2_contingency(word_label, correction=True)[1]

    if p_value < 0.1:
        res_cond1 = True
    return res_cond1


def judgeCon2(vec_word_gender0, vec_word_gender1):
    res_cond2 = False
    '''
    k-s 检验, only for continuous distribution
    '''
    # res = stats.kstest(vec_word_gender0,vec_word_gender1)
    # p_value = res.pvalue
    # if p_value<0.1: #拒绝原假设(X|T=0，X|T=1 分布相同)
    #     res_cond2 = True
    '''
    二项分布检验
    '''
    p = np.sum(vec_word_gender1) / vec_word_gender1.shape[0]
    k = np.sum(vec_word_gender0)
    n = vec_word_gender0.shape[0]
    p_value = stats.binomtest(k, n=n, p=p, alternative='two-sided').pvalue  # "greater"

    if p_value < 0.1:  # 拒绝原假设(X|T=0，X|T=1 分布相同)
        res_cond2 = True
    return res_cond2


def get_T(data_name, vec, oneT):
    sentences, label = getTrainSenLabel(data_name)
    label = label.numpy()
    label = np.where(label == 1, label, -1 * np.ones_like(label))
    vec_sens = np.asarray(vec.transform(sentences).todense())
    vec_dic = vec.vocabulary_
    if data_name == "AMI":
        id_lst = [["woman", "man"], ["women", "men"], ["girls", "boys"], ["girl", "boy"],
                  ["she", "he"],
                  ["wife", "husband"],
                  ["lady", "gentleman"],
                  ["ladies", "gentlemen"],
                  ["girlfriend", "boyfriend"],
                  ["sister", "brother"], ["mother", "father"], ["daughter", "son"],
                  ["gal", "guy"],
                  ["female", "male"],
                  ["her", "his"],
                  ["herself", "himself"]]
        sensitive_words = [x[0] for x in id_lst]
        sensitive_words = list(filter(lambda x: x in list(vec.vocabulary_.keys()), sensitive_words))
        if oneT:
            sensitive_words = [sensitive_words]
        else:
            sensitive_words = [[x] for x in sensitive_words]
    # elif data_name in ["IMDB-S", "IMDB-L"]:
    #     domain_words = ['movie', 'film', 'story', 'movies', 'films',
    #                 'performance', 'performances', 'script', 'actor', 'actress',
    #                 'cast', 'acting', 'actors', 'actresses', 'action', 'played',
    #                 'characters', 'character', 'scenes', 'scene', 'plot',
    #                 'role', 'music', 'director', 'line', 'plots', 'casting',
    #                 'acted', 'play', 'plays', 'lines'
    #                 ]
    #     sensitive_words = []
    #     for w in tqdm(domain_words):
    #         vec_word = vec_sens[:, vec_dic[w]]
    #         flag = judgeCon1(vec_word, label)
    #         if flag:
    #             sensitive_words.append(w)
    #     if one_T:
    #         sensitive_words = [sensitive_words]
    #     else:
    #         sensitive_words = [[x] for x in sensitive_words]

    elif data_name == "IMDB-S":
        ds = load_pkl(Config.DATA_DIC[data_name])
        causal_words = ds.all_causal_terms.term.values.tolist()
        stop_lst = stopwords.words('english')
        sensitive_words = []
        for w in tqdm(list(vec.vocabulary_.keys())):
            if (w not in causal_words) and (w not in stop_lst):
                vec_word = vec_sens[:, vec_dic[w]]
                if np.sum(vec_word) < 50:
                    flag = False
                else:
                    flag = True
                # if np.sum(vec_word) < 50:
                #     flag = False
                # else:
                #     flag = judgeCon1(vec_word, label)
                if flag:
                    sensitive_words.append(w)
        if oneT:
            sensitive_words = [sensitive_words]
        else:
            sensitive_words = word_cluster(sensitive_words)
    elif data_name == "WASEEM":
        ds = load_pkl(Config.DATA_DIC[data_name])
        sensitive_words = ds.sensitive_words
        sensitive_words = list(filter(lambda x: x in list(vec.vocabulary_.keys()), sensitive_words))
        if oneT:
            sensitive_words = [sensitive_words]
        else:
            sensitive_words = [[x] for x in sensitive_words]
    return sensitive_words


def get_X_for_Ts(vec, gender_words):
    covariates = []
    for word in list(vec.vocabulary_.keys()):
        if word not in gender_words:
            covariates.append(word)
    return covariates


def getT0T1ROW(vocab_gender_words, vec_sens, vec_dic):
    gender_id = [vec_dic[w] for w in vocab_gender_words]

    vec_sens_gender = vec_sens[:, [gender_id]].reshape(vec_sens.shape[0], -1)
    vec_sens_gender_sum = np.sum(vec_sens_gender, axis=-1)

    gender0_raw = np.where(vec_sens_gender_sum == 0)[0]  # (num_samples_gender0,)
    gender1_raw = np.where(vec_sens_gender_sum != 0)[0]  # (num_samples_gender1,)

    return gender0_raw, gender1_raw


def get_cfder_wd(gender0_raw, gender1_raw, covariates, vec_sens, label, vec_dic):
    vec_sens_gender0 = vec_sens[gender0_raw, :]  # (num_samples_gender0, num_vocab)
    label_gender0 = label[gender0_raw]  # (num_samples_gender0,)
    vec_sens_gender1 = vec_sens[gender1_raw, :]  # (num_samples_gender1, num_vocab)
    label_gender1 = label[gender1_raw]  # (num_samples_gender1,)

    cond1 = []
    cond2 = []
    confouder_for_gender = []
    for word in tqdm(covariates):
        # word = "bitch"
        vec_word_gender0 = vec_sens_gender0[:, vec_dic[word]]
        vec_word_gender1 = vec_sens_gender1[:, vec_dic[word]]
        try:
            res1 = judgeCon1(vec_word_gender0, label_gender0)
            res2 = judgeCon2(vec_word_gender0, vec_word_gender1)
        except:
            continue
        cond1.append(res1)
        cond2.append(res2)
        if res1 and res2:
            confouder_for_gender.append(word)

    return confouder_for_gender


def train_weights(all_XT, all_XC, all_gender0_row, all_gender1_row, sample_num):
    model = WeightLinear(sample_num=sample_num)
    optimizer = AdamW(model.parameters(),
                      lr=1e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
                      weight_decay=0.001)
    if model.device == torch.device("cuda"):
        model.cuda()
    model.train()
    loss_train = []
    for i in trange(50000):
        model.zero_grad()
        loss_allt = 0
        # loss_weight_allt = 0
        # loss_weight_reg_allt = 0
        for XC, XT, gender0_row, gender1_row in zip(all_XC, all_XT, all_gender0_row, all_gender1_row):
            XC = XC.to(model.device)
            XT = XT.to(model.device)
            gender0_row = torch.as_tensor(gender0_row).to(model.device)
            gender1_row = torch.as_tensor(gender1_row).to(model.device)

            loss_dic = model(XC, XT, gender0_row, gender1_row)
            # print(loss_dic)
            loss_allt += loss_dic["loss"]
            # loss_weight_allt+=loss_dic["weight_loss"]
            # loss_weight_reg_allt+=loss_dic["weight_reg_loss"]

        loss_allt.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f'\t\t{i // 20}th 20 step loss:', loss_allt.item(),
                  # loss_weight_allt.item(),loss_weight_reg_allt.item()
                  )
            # break
            # if i >40:
            #     delta = loss_train[-1]-loss_allt
            #     if delta < 1e-8:
            #         break
        loss_train.append(loss_allt)

    for params in model.named_parameters():
        [name, param] = params
        if param.grad is not None:
            weight_all = param.data
            # print(name, end='\t')
            # print('weight:{}'.format(param.data), end='\t')
            # print(torch.argmax(param.data))
            # print('grad:{}'.format(param.grad.data.mean()))

    weight_all = torch.exp(weight_all)  # if t samples with weight 1/cfder_sens_gender1.shape(0)

    return weight_all


def make_weights(all_cfder, all_gender0_row, all_gender1_row, vec_sens, vec_dic):
    sample_num = vec_sens.shape[0]
    all_XT = []
    all_XC = []
    for confouder_for_gender, gender0_row, gender1_row in zip(all_cfder,
                                                              all_gender0_row,
                                                              all_gender1_row):
        cfder_ids = [vec_dic[cfd] for cfd in confouder_for_gender]
        cfder_sens = vec_sens[:, cfder_ids]

        cfder_sens_gender1 = cfder_sens[gender1_row, :]
        cfder_sens_gender0 = cfder_sens[gender0_row, :]

        # input
        # xt_avg = torch.as_tensor(np.average(cfder_sens_gender1,axis=0), dtype=torch.float)
        XC = torch.as_tensor(cfder_sens_gender0, dtype=torch.float)
        XT = torch.as_tensor(cfder_sens_gender1, dtype=torch.float)
        all_XT.append(XT)
        all_XC.append(XC)
        # print(XC.size())  # torch.Size([2777, 78])
        # print(xt_avg.size())  # torch.Size([78])

    weight_all = train_weights(all_XT, all_XC, all_gender0_row, all_gender1_row, sample_num)

    weight_all = weight_all.cpu().numpy()
    # weight_all = sample_num * weight_all
    # weight_c = weight_c.numpy() * cfder_sens_gender1.shape[0]  # if t samples with weight 1
    #
    # weight_all =np.ones(cfder_sens_gender1.shape[0]+cfder_sens_gender0.shape[0])
    # weight_all[gender0_raw] = weight_c

    return weight_all


def pipline(data_name, oneT=False):
    # read the data

    sentences, label = getTrainSenLabel(data_name)
    label = label.numpy()
    label = np.where(label == 1, label, -1 * np.ones_like(label))

    # make the vocab
    vec = CountVectorizer(min_df=3, binary=True, max_df=.8, stop_words="english")
    vec.fit(sentences)
    vec_sens = np.asarray(vec.transform(sentences).todense())
    vec_dic = vec.vocabulary_

    # get T words and it's covariates
    all_vocab_gender_words = get_T(data_name, vec, oneT)

    all_cfder = []
    all_gender0_row = []
    all_gender1_row = []
    for i, vocab_gender_words in enumerate(tqdm(all_vocab_gender_words)):
        print(i)
        covariates = get_X_for_Ts(vec, vocab_gender_words)

        # detect the cofounder
        gender0_raw, gender1_raw = getT0T1ROW(vocab_gender_words, vec_sens, vec_dic)
        confouder_for_gender = get_cfder_wd(gender0_raw, gender1_raw, covariates, vec_sens, label, vec_dic)

        all_cfder.append(confouder_for_gender)
        all_gender0_row.append(gender0_raw)
        all_gender1_row.append(gender1_raw)

    # learn the weight tp balance the cofounder
    weight_all = make_weights(all_cfder, all_gender0_row, all_gender1_row, vec_sens, vec_dic)

    # save the data
    ds = load_pkl(Config.DATA_DIC[data_name])
    # ds.sensitive_words = all_vocab_gender_words
    ds.sensi_cfder = all_cfder
    ds.train["weight_sum"] = weight_all.tolist()
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def check(data_name, word):
    data_name = "AMI"
    ds = load_pkl(Config.DATA_DIC[data_name])
    weight_all = ds.train["weight_sum"]

    sentences, label = getTrainSenLabel(data_name)
    label = label.numpy()
    label = np.where(label == 1, label, -1 * np.ones_like(label))

    # make the vocab
    vec = CountVectorizer(min_df=3, binary=True, max_df=.8, stop_words="english")
    vec.fit(sentences)
    vec_sens = np.asarray(vec.transform(sentences).todense())
    vec_dic = vec.vocabulary_

    vocab_gender_words = ds.sensitive_words[0]
    cfder_words = ds.sensi_cfder[0]
    # cfder_words = all_cfder[0]

    gender0_raw, gender1_raw = getT0T1ROW(vocab_gender_words, vec_sens, vec_dic)

    # vec_sens_gender0 = vec_sens[gender0_raw, :]  # (num_samples_gender0, num_vocab)
    label_gender0 = label[gender0_raw]  # (num_samples_gender0,)
    # vec_sens_gender1 = vec_sens[gender1_raw, :]  # (num_samples_gender1, num_vocab)
    label_gender1 = label[gender1_raw]  # (num_samples_gender1,)
    print("cfder", " overall", " gender=1", " gender=0")
    for cfder in cfder_words:
        vec_cfder = vec_sens[:, vec_dic[cfder]]
        vec_cfder_gender0 = vec_cfder[gender0_raw]
        vec_cfder_gender1 = vec_cfder[gender1_raw]

        print(cfder, np.sum(vec_cfder) / 4000,
              np.sum(vec_cfder_gender1) / gender1_raw.shape[0],
              np.sum(vec_cfder_gender0) / gender0_raw.shape[0])

        vec_bitch_weighted = weight_all * vec_cfder

        vec_bitch_gender0_weighted = vec_bitch_weighted[gender0_raw]
        vec_bitch_gender1_weighted = vec_bitch_weighted[gender1_raw]

        print(cfder, np.sum(vec_bitch_weighted) / 4000,
              np.sum(vec_bitch_gender1_weighted) / gender1_raw.shape[0],
              np.sum(vec_bitch_gender0_weighted) / gender0_raw.shape[0])


data_name = "AMI"
oneT = True
pipline(data_name, oneT)