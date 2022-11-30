from scipy import stats
from utils.funcs import getTrainSenLabel, getUnbiasedSen
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
from utils.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class WeightLinear(nn.Module):
    def __init__(self, gender0_num=2777):
        super(WeightLinear, self).__init__()
        # self.w = nn.Parameter(1/1/torch.ones(gender0_num,), requires_grad=True)
        self.gender0_num = gender0_num
        self.w = nn.Parameter(torch.zeros(gender0_num,),requires_grad = True )

    def forward(self, XC, xt_avg):

        # print(XC.size())  # torch.Size([2777, 78])
        # print(xt_avg.size())  # torch.Size([78])
        # print(self.w.size())


        xc_avg = torch.matmul(XC.t(), torch.exp(self.w.unsqueeze(-1))/self.gender0_num)



        weight_loss = torch.nn.functional.mse_loss(input=xc_avg.squeeze(-1),
                                                   target=xt_avg,
                                                   reduction='sum')
        return weight_loss


def load_pkl(path):
    pickle_file = open(path,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data


def judgeCon1(vec_word_gender0, label_gender0):
    '''

    :param vec_word_gender0: （T=0的样本数目，），每一行代表这个样本句子是否包含word，0/1
    :param label_gender0: （T=0的样本数目，），每一行代表这个样本的分类标签，0/1
    :return:
    # 这个条件衡量的是word 对X有没有直接影响能力， 以T=0为条件，是为了排除word与Y的相关性不是混杂u带来的(word<--u-->Y)
    '''
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

    if p_value<0.1:
        res_cond1 = True
    return res_cond1


def judgeCon2(vec_word_gender0, vec_word_gender1):
    '''
    :param vec_word_gender0: （T=0的样本数目，），每一行代表这个样本句子是否包含word，0/1
    :param vec_word_gender1: （T=1的样本数目，），每一行代表这个样本句子是否包含word，0/1
    :return:
    # 这个条件衡量的是 X|T=0, X|T=1 分布是否相同。
    # 也就是说衡量单词X对T有没有倾向性
    '''
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
    p = np.sum(vec_word_gender1)/vec_word_gender1.shape[0]
    k = np.sum(vec_word_gender0)
    n = vec_word_gender0.shape[0]
    p_value= stats.binomtest(k, n=n, p=p, alternative='greater').pvalue

    if p_value<0.1: #拒绝原假设(X|T=0，X|T=1 分布相同)
        res_cond2 = True
    return res_cond2



def get_cfder(covariates,vec_sens_gender0,vec_sens_gender1,vec_dic,label_gender0):

    cond1 = []
    cond2 = []
    confouder_for_gender = []

    for word in tqdm(covariates):
        # word = "friend"
        vec_word_gender0 = vec_sens_gender0[:,vec_dic[word]]
        vec_word_gender1 = vec_sens_gender1[:,vec_dic[word]]
        if np.sum(vec_word_gender0)>20 and np.sum(vec_word_gender1)>20:
            try:
                res1 = judgeCon1(vec_word_gender0, label_gender0)
                res2 = judgeCon2(vec_word_gender0, vec_word_gender1)
            except:
                continue
            cond1.append(res1)
            cond2.append(res2)
            if res1 and res2:
                confouder_for_gender.append(word)
        else:
            continue
    return confouder_for_gender


def pipeline(data_name):
    data_name = "AMI"

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
    gender_words = [x[0] for x in id_lst]


    sentences, label = getTrainSenLabel(data_name)
    label = label.numpy()
    label = np.where(label == 1, label, -1 * np.ones_like(label))

    vec = CountVectorizer(min_df=3, binary=True, max_df=.8, stop_words="english")
    vec.fit(sentences)
    vec_sens = np.asarray(vec.transform(sentences).todense())
    vec_dic = vec.vocabulary_

    covariates = []
    vocab_gender_words = []
    for word in list(vec.vocabulary_.keys()):
        if word not in gender_words:
            covariates.append(word)
        else:
            vocab_gender_words.append(word)



    gender_id = [vec_dic[w] for w in vocab_gender_words]

    vec_sens_gender = vec_sens[:, [gender_id]].reshape(vec_sens.shape[0], -1)
    vec_sens_gender_sum = np.sum(vec_sens_gender, axis=-1)

    gender0_raw = np.where(vec_sens_gender_sum == 0)[0]  # (num_samples_gender0,)
    vec_sens_gender0 = vec_sens[gender0_raw, :]  # (num_samples_gender0, num_vocab)
    label_gender0 = label[gender0_raw]  # (num_samples_gender0,)

    gender1_raw = np.where(vec_sens_gender_sum != 0)[0]  # (num_samples_gender1,)
    vec_sens_gender1 = vec_sens[gender1_raw, :]  # (num_samples_gender1, num_vocab)
    label_gender1 = label[gender1_raw]  # (num_samples_gender1,)

    confouder_for_gender = get_cfder(covariates, vec_sens_gender0, vec_sens_gender1, vec_dic, label_gender0)


    cfder_ids = [vec_dic[cfd] for cfd in confouder_for_gender]
    cfder_sens = vec_sens[:,cfder_ids]
    cfder_sens_gender1 = cfder_sens[gender1_raw, :]
    cfder_sens_gender0 = cfder_sens[gender0_raw, :]

    xt_avg = torch.as_tensor(np.average(cfder_sens_gender1,axis=0), dtype=torch.float)

    XC = torch.as_tensor(cfder_sens_gender0, dtype=torch.float)

    gender0_num = XC.size()[0]
    model = WeightLinear(gender0_num=gender0_num)
    optimizer = AdamW(model.parameters(),
                          lr=1e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
                          weight_decay=0.001)
    model.train()
    loss_train = []
    for i in trange(50000):
        model.zero_grad()
        loss = model(XC, xt_avg)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f'\t\t{i // 20}th 20 step loss:', loss)
        loss_train.append(loss)

    for params in model.named_parameters():

        [name, param] = params
        if param.grad is not None:
            weight_avg = param.data
            # print(name, end='\t')
            # print('weight:{}'.format(param.data), end='\t')
            # print(torch.argmax(param.data))
            # print('grad:{}'.format(param.grad.data.mean()))

    weight_c = torch.exp(weight_avg)/model.gender0_num  # if t samples with weight 1/cfder_sens_gender1.shape(0)
    weight_c = weight_c.numpy() * cfder_sens_gender1.shape[0]  # if t samples with weight 1

    weight_all =np.ones(cfder_sens_gender1.shape[0]+cfder_sens_gender0.shape[0])
    weight_all[gender0_raw] = weight_c
    ds = load_pkl(Config.DATA_DIC[data_name])
    ds.train["weight"] = weight_all.tolist()
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))


def TopK(model, top_k, vectorizer):
    coef = model.coef_[0]  # ndarray
    vocab_dic = vectorizer.vocabulary_
    coef_dic = {}
    top_k_words = []
    for k, v in vocab_dic.items():
        coef_dic[k] = coef[v]
    for k, v in sorted(coef_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top_k]:
        top_k_words.append((k,v))
    return top_k_words


def weight_train_lg(data_name):
    data_name = "AMI"
    ds = load_pkl(Config.DATA_DIC[data_name])
    sentences, label = getTrainSenLabel(data_name)
    vec = CountVectorizer(min_df=3, binary=True, max_df=.8, stop_words="english")
    vec.fit(sentences)
    vec_train = vec.transform(sentences)
    vec_train = np.asarray(vec_train.todense())


    label = label.numpy()

    model = LogisticRegression(solver='sag', verbose=2)
    model.fit(vec_train, label)

    weighted_model = LogisticRegression(solver='sag', verbose=2)
    weighted_model.fit(vec_train, label,sample_weight=ds.train.weight.values.tolist())

    top_k_words = TopK(model,20,vec)
    top_k_words_weight = TopK(weighted_model,20,vec)

    print("top_k_words:",top_k_words)
    print("top_k_words_weight:",top_k_words_weight)

    test_sentences, df = getUnbiasedSen(data_name)
    if "label" in df.columns:
        df.loc[df["label"] == -1, "label"] = 0
        print(df["label"].value_counts())
        y_test = df["label"].tolist()

    elif "misogynous" in df.columns:
        y_test = df["misogynous"].tolist()

    vec_test = vec.transform(test_sentences)

    y_dev_pred_word = model.predict(vec_test)
    print("test", accuracy_score(y_test, y_dev_pred_word)) #52

    y_dev_pred_word = weighted_model.predict(vec_test)
    print("test", accuracy_score(y_test, y_dev_pred_word)) #51



