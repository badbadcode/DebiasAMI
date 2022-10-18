'''
[1] Shen Z ,  Cui P ,  Kuang K , et al. Causally Regularized Learning with Agnostic Data Selection Bias[J].  2017.
'''
import os
import pandas as pd
import torch
from datasets import Dataset
from utils.config import Config
import pickle
import numpy as np
import random
from utils.models import BertFT
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from train_lm import mask_all,mask_i


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

def getModel(model_shortcut,data_name, seed):

    if model_shortcut == "b-ft":
        model = BertFT(data_name=data_name, seed=seed)
    if model.device == torch.device("cuda"):
        model.cuda()
    # if model_shortcut == "bilstm":
    #     model = SentiBiLSTMAtt(n_vocab=Config.vocab_size[Config.tokenizer_name], num_labels=Config.NUM_LABELS[train_data])
    # elif model_shortcut == "cbilstm":
    #     model = CausalSentiBiLSTMAtt(n_vocab=Config.vocab_size[Config.tokenizer_name], num_labels=Config.NUM_LABELS[train_data],mode=Config.MODE[model_shortcut])
    #     # print("model.causal_t.Wy.weight", model.causal_t.Wy.weight)  # changed when it wants to change
    # elif model_shortcut == "b":
    #     model = SentiBERT(num_labels=Config.NUM_LABELS[train_data])
    # elif model_shortcut == "cb":
    #     model = CausalSentiBERT(num_labels=Config.NUM_LABELS[train_data], mode=Config.MODE[model_shortcut])
    #     # print("model.causal_t.Wy.weight", model.causal_t.Wy.weight)  # changed when it wants to change
    # elif model_shortcut == "panet":
    #     model = PANet(n_vocab=Config.vocab_size[Config.tokenizer_name], num_labels=Config.NUM_LABELS[train_data])

    return model

def getTrainDevTensor(sentences, labels, model):
    tokenizer = AutoTokenizer.from_pretrained(model.model_name, do_lower_case=True)
    # print("sentences", len(sentences),sentences[:5])
    encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=model.max_len)
    input_ids = torch.tensor(encoded_inputs["input_ids"])
    attention_masks = torch.tensor(encoded_inputs["attention_mask"])

    tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
    return tensor_dataset

def getMaskedInput(model, data_name):
    data_fp = Config.DATA_DIC[data_name]["train"]
    df = pd.read_csv(data_fp, sep="\t", header=0)
    sentences = list(df['cleaned_text'].values.astype('U'))
    # print(sentences[:5])
    labels = torch.tensor(df["misogynous"].tolist())
    tokenizer = AutoTokenizer.from_pretrained(model.model_name, do_lower_case=True)
    # print("sentences", len(sentences),sentences[:5])
    encoded_inputs = tokenizer(sentences, padding='max_length',truncation=True, max_length=model.max_len)
    input_ids = encoded_inputs["input_ids"]
    attention_masks = encoded_inputs["attention_mask"]

    new_input_ids,new_attention_masks = mask_all(input_ids,attention_masks) #[num_samples, seq_len-2, seq_len]
    new_labels = [[label]*len(id) for id,label in zip(input_ids,labels)]

    # flat_input_ids = sum(new_input_ids,[])
    # flat_attention_masks = sum(new_attention_masks, [])
    # flat_labels = sum(new_labels, [])

    # tensor_dataset = TensorDataset(new_input_ids, new_attention_masks,new_labels)

    return new_input_ids, new_attention_masks,new_labels



def getTrainDevLoader(model,data_name):

    data_fp = Config.DATA_DIC[data_name]["train"]
    df = pd.read_csv(data_fp, sep="\t", header=0)
    sentences = list(df['cleaned_text'].values.astype('U'))
    # print(sentences[:5])
    labels = torch.tensor(df["misogynous"].tolist())

    Sen_train, Sen_dev, Lbl_train, Lbl_dev = train_test_split(sentences, labels, test_size=0.2)
    # print(Sen_dev[:5])
    train_data = getTrainDevTensor(Sen_train, Lbl_train, model)
    dev_data = getTrainDevTensor(Sen_dev, Lbl_dev, model)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=model.batch_size)
    dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=model.batch_size)

    return train_dataloader, dev_dataloader


def getTestLoader(model, data_name, test_name):

    Has_label = Config.HAS_LABELS_TEST[data_name][test_name]

    data_fp = Config.DATA_DIC[data_name][test_name]
    df = pd.read_csv(data_fp, sep="\t", header=0)
    sentences = list(df['cleaned_text'].values.astype('U'))

    tokenizer = AutoTokenizer.from_pretrained(model.model_name, do_lower_case=True)
    encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=model.max_len)
    input_ids = torch.tensor(encoded_inputs["input_ids"])
    attention_masks = torch.tensor(encoded_inputs["attention_mask"])

    if Has_label:
        labels = torch.tensor(df["misogynous"].tolist())
        test_data = TensorDataset(input_ids, attention_masks, labels)
    else:
        test_data = TensorDataset(input_ids, attention_masks)

    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=model.batch_size)

    return test_dataloader


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
        if fem_index[j] == 1 :
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


def get_sentence_vecs(vec_type, path=None):
    '''
    获得 aclImdb数据级经过Bert编码后的句向量

    :param vec_type: 向量类型 包括['last_hidden_state_first', 'pooler_output']
    :return: train_dataset, test_dataset
    '''

    path = "data/AMI EVALITA 2018/vector" if path is None else path
    train_X = torch.load(os.path.join(path, vec_type, 'train.pt'))
    data_train = pd.read_csv(Config.cleaned_train_fp, sep="\t", header=0)
    train_Y = torch.LongTensor(data_train['misogynous'].values.astype('int').tolist())

    test_X = torch.load(os.path.join(path, vec_type, f'test.pt'))
    data_test = pd.read_csv(Config.cleaned_test_fp, sep="\t", header=0)
    test_Y = torch.LongTensor(data_test['misogynous'].values.astype('int').tolist())

    test_fair_X = torch.load(os.path.join(path, vec_type, f'fair.pt'))
    data_test_fair = pd.read_csv(Config.cleaned_test_fair_fp, sep="\t", header=0)
    test_fair_Y = torch.LongTensor(data_test_fair['misogynous'].values.astype('int').tolist())

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

def getGenderIndex(vec_type, only_female=True):
    '''
    :param vec_type: 编码的类型
    :param only_female: 是否将所有单词都看做是处置变量
    :return:
    '''
    if only_female:
        baseline_swap_list = [["woman", "man"],["girls","boys"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"]]
        female_words = [x[0] for x in baseline_swap_list]
        if vec_type in ["onehot", "tfidf"]:
            vectorizer_path = f"{Config.vectorizer_path}{vec_type}_vectorizer.pkl"
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
            vectorizer_path = f"{Config.vectorizer_path}{vec_type}_vectorizer.pkl"
            vectorizer = pickle.load(open(vectorizer_path, "rb"))
            vocab_dic = vectorizer.vocabulary_
            gender_index = np.ones(len(vocab_dic.keys()))
        else:
            gender_index = np.ones(768)

    return gender_index


if __name__ == '__main__':
    seed = 824
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    X = torch.randn(100000, 5)
    Y = torch.zeros(X.shape[0])
    Y[torch.rand_like(Y) > 0.5] = 1
    pred = torch.rand(X.shape[0])
    I = torch.zeros_like(X)
    I[torch.rand(*X.shape) > 0.5] = 1
    w = torch.rand(X.shape[0])

    losses = []
    lr = 1e-3
    lr_decay = 0.5
    lambda1 = 1e-2
    lambda2 = 1e-3
    lambda5 = 1e-8
    for i in range(10):
        w, loss, loss_detail = update_w_one_step(X, Y, w, I, pred, lambda1, lambda2, lambda5, lr, device='cpu')
        losses.append(loss)
        lr = lr * lr_decay

    print(losses[:3], losses[-3:])
    print((w * w)[:10])
    print((w * w).sum())

