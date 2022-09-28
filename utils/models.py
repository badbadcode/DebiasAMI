import torch
import torch.nn as nn
from config import Config
from transformers import AutoTokenizer
import pickle

class Vector_NN_Classifier(nn.Module):
    #类变量
    causal_hyper = None

    #实例方法
    def __init__(self, hidden_size, num_labels):
        super(Vector_NN_Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.classifier = nn.Sequential(nn.Linear(hidden_size, num_labels))

        self.loss_f = nn.CrossEntropyLoss() #这里似乎并没有用到，是直接写出来的交叉熵函数

    def forward(self, x, x_indexes, labels=None, mode='train'):
        # 初始化的分类器就是一个全连接神经网络
        # print("x", x.size()) #100,882

        x = self.classifier(x)
        prediction = torch.softmax(x, dim=1)

        if mode == 'train':
            if Vector_NN_Classifier.causal_hyper is None:  #这一步基本是这个类实例化对象之后，必须要设置因果超参数，于是这个类也写了一个静态函数（设置超参数）
                raise ValueError('请设置超参')
            # 这里已经设置了因果超参数这个类变量（字典类型）
            w = Vector_NN_Classifier.causal_hyper['w']
            lambda3 = Vector_NN_Classifier.causal_hyper['lambdas'][3]
            lambda4 = Vector_NN_Classifier.causal_hyper['lambdas'][4]

            #依次计算这个模型中参数的正则L1，L2
            L1 = 0
            L2 = 0
            for param in self.parameters():
                L1 += param.abs().sum()
                L2 += (param ** 2).sum()


            W = w * w #为了保证权重非负

            W_selected = torch.gather(W, 0, x_indexes) #sample weight 有什么必要?

            # log_expYX = (1 - labels) * prediction[:, 0] + (labels) * prediction[:, 1]
            # log_expYX = torch.log(1 + torch.exp(log_expYX))
            # part1 = Variable(W_selected * log_expYX, requires_grad=True)
            # part1 = part1.sum()

            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1

            part1 = W_selected * part1
            part1 = part1.sum()

            # print('loss:', loss)
            loss = part1 + lambda3 * L2 + lambda4 * L1 #这里是为了更新分类模型参数的loss，也就是说论文中的公式5
            # print('lambda3:', lambda3, 'lambda4:', lambda4)
            # print('L2:', L2, 'L1:', L1)
            return {
                'loss': loss,
                'loss_detail': torch.tensor([part1, lambda3 * L2, lambda4 * L1]),
                'prediction': prediction.argmax(1),
                'pred_orign': prediction
            }
        elif mode == 'eval':#没有加权
            lambda3 = Vector_NN_Classifier.causal_hyper['lambdas'][3]
            lambda4 = Vector_NN_Classifier.causal_hyper['lambdas'][4]

            L1 = 0
            L2 = 0
            for param in self.parameters():
                L1 += param.abs().sum()
                L2 += (param ** 2).sum()

            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1
            part1 = part1.sum()

            # print('loss:', loss)
            loss = part1 + lambda3 * L2 + lambda4 * L1
            return {
                'prediction': prediction.argmax(1),
                'loss': loss
            }

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions

        return {
            'acc': (labels == preds).sum() / len(labels)
        }

    @staticmethod
    def set_causal_hyperparameter(w, lambdas):
        Vector_NN_Classifier.causal_hyper = {
            'w': w,
            'lambdas': lambdas
        }




class Vector_LogisticNet_Classifier(torch.nn.Module):

    #类变量
    causal_hyper = None

    def __init__(self, hidden_size,num_labels):
        super(Vector_LogisticNet_Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.linear = torch.nn.Linear(hidden_size, num_labels)


    # 向前传播
    def forward(self, x, x_indexes, labels=None, mode="train"):

        prediction_1 = 1/(1+torch.exp(-1*self.linear(x)))
        prediction_0 = torch.ones_like(prediction_1) - prediction_1
        prediction = torch.cat((prediction_0,prediction_1),dim=1)

        if mode == 'train':
            if Vector_LogisticNet_Classifier.causal_hyper is None:  #这一步基本是这个类实例化对象之后，必须要设置因果超参数，于是这个类也写了一个静态函数（设置超参数）
                raise ValueError('请设置超参')
            # 这里已经设置了因果超参数这个类变量（字典类型）
            w = Vector_LogisticNet_Classifier.causal_hyper['w']
            lambda3 = Vector_LogisticNet_Classifier.causal_hyper['lambdas'][3]
            lambda4 = Vector_LogisticNet_Classifier.causal_hyper['lambdas'][4]

            #依次计算这个模型中参数的正则L1，L2
            L1 = 0
            L2 = 0
            for param in self.parameters():
                L1 += param.abs().sum()
                L2 += (param ** 2).sum()


            W = w * w #为了保证权重非负

            W_selected = torch.gather(W, 0, x_indexes) #sample weight 有什么必要?

            # log_expYX = (1 - labels) * prediction[:, 0] + (labels) * prediction[:, 1]
            # log_expYX = torch.log(1 + torch.exp(log_expYX))
            # part1 = Variable(W_selected * log_expYX, requires_grad=True)
            # part1 = part1.sum()

            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1

            part1 = W_selected * part1
            part1 = part1.sum()

            # print('loss:', loss)
            loss = part1 + lambda3 * L2 + lambda4 * L1 #这里是为了更新分类模型参数的loss，也就是说论文中的公式5
            # print('lambda3:', lambda3, 'lambda4:', lambda4)
            # print('L2:', L2, 'L1:', L1)
            return {
                'loss': loss,
                'loss_detail': torch.tensor([part1, L2, L1]),
                'prediction': prediction.argmax(1),
                'pred_orign': prediction
            }
        elif mode == 'eval':
            lambda3 = Vector_LogisticNet_Classifier.causal_hyper['lambdas'][3]
            lambda4 = Vector_LogisticNet_Classifier.causal_hyper['lambdas'][4]

            L1 = 0
            L2 = 0
            for param in self.parameters():
                L1 += param.abs().sum()
                L2 += (param ** 2).sum()
            # print(prediction)
            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1
            part1 = part1.sum()

            # print('loss:', loss)
            loss = part1 + lambda3 * L2 + lambda4 * L1
            return {
                'prediction': prediction.argmax(1),
                'loss': loss
            }


    @staticmethod
    def set_causal_hyperparameter(w, lambdas):
        Vector_LogisticNet_Classifier.causal_hyper = {
            'w': w,
            'lambdas': lambdas
        }

    @staticmethod
    def TopK(model,vec_type):
        coef = torch.squeeze(model["linear.weight"],0).numpy()
        # print(coef)
        if vec_type == "bert":
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            vocab_dic = tokenizer.get_vocab()
        else:
            vectorizer_path = f"{Config.vectorizer_path}{vec_type}_vectorizer.pkl"
            vectorizer = pickle.load(open(vectorizer_path, "rb"))
            vocab_dic = vectorizer.vocabulary_

        coef_dic = {}

        for k, v in vocab_dic.items():
            coef_dic[k] = coef[v]

        sorted_coef_dic = sorted(coef_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        return sorted_coef_dic




