import torch
import torch.nn as nn
from utils.config import Config
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import pickle
import numpy as np
import torch.nn.functional as F

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

            # print("x_indexes", x_indexes)
            W_selected = torch.gather(W, 0, x_indexes) #sample weight 选择这个batch里的样本应有的权重
            # print("W_selected",W_selected)
            # log_expYX = (1 - labels) * prediction[:, 0] + (labels) * prediction[:, 1]
            # log_expYX = torch.log(1 + torch.exp(log_expYX))
            # part1 = Variable(W_selected * log_expYX, requires_grad=True)
            # part1 = part1.sum()

            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1
            # print("part1",part1)
            part1 = W_selected * part1

            part1 = part1.sum()
            # print("sumed part1",part1)

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
            # print("x_indexes", x_indexes)
            W_selected = torch.gather(W, 0, x_indexes) #sample weight 有什么必要?
            # print("W_selected", W_selected)
            # log_expYX = (1 - labels) * prediction[:, 0] + (labels) * prediction[:, 1]
            # log_expYX = torch.log(1 + torch.exp(log_expYX))
            # part1 = Variable(W_selected * log_expYX, requires_grad=True)
            # part1 = part1.sum()

            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1
            # print("part1", part1)

            part1 = W_selected * part1
            part1 = part1.sum()
            # print("sum_part1", part1)
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

class BertFT(nn.Module):
    def __init__(self, data_name, seed, mode):
        super(BertFT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=Config.NUM_LABELS[data_name])
        self.ffn = nn.Linear(self.bert.config.hidden_size, Config.NUM_LABELS[data_name])

        self.model_name = "bert-base-uncased"
        self.mode = mode  # mask or normal
        self.model_shortcut = "b-ft"
        self.data_name = data_name
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 64
        self.lr = 1e-5
        self.weight_decay = 0.001
        self.warm_up = 0.1
        self.patience = 3
        self.epochs = 20
        self.max_len = 80
        self.seed = seed
        self.save_model_path = f"saved_model/{data_name}/{self.model_shortcut}_{str(self.lr)}_{str(seed)}_{mode}.pt"

    def forward(self, input_ids, input_mask, labels=None, haslabel=True):
        outputs = self.bert(input_ids,
                             token_type_ids=None,
                             attention_mask= input_mask, output_hidden_states=True)
        # print("\n ==output=====")
        # print(len(outputs.hidden_states)) #13 the first one is embedding
        z0 = outputs.hidden_states[-1][:, 0, :]  # the hidden_states of [CLS]

        logits = self.ffn(z0)

        # probs = nn.functional.softmax(logits,dim=1)
        # print(probs.size())  # torch.Size([28, 2])
        # print(labels.size())  # torch.Size([28])
        if haslabel:
            loss = nn.functional.cross_entropy(logits, labels)
            return {"logits":nn.functional.softmax(logits,dim=1),
                    "prediction":logits.argmax(1),
                    "loss":loss,
                    "label":labels}
        else:
            return {"logits":nn.functional.softmax(logits,dim=1),
                    "prediction":logits.argmax(1)
                    }

class BiGRU(nn.Module):
    def __init__(self, data_name, hidden_size):
        super(BiGRU, self).__init__()

        self.dropout =0.2
        self.hidden_size = hidden_size
        self.embed_size = 300
        self.num_layers = 1
        # self.embedding = EmbedBERT()
        ds = pickle.load(open(Config.DATA_DIC[data_name],'rb'))

        word_embed = ds.train_embed_matrix #(n_vocab, dim)
        #vocab_size, embedding_dim
        word_embed = word_embed.to(torch.float32)
        self.embedding = nn.Embedding.from_pretrained(word_embed,freeze=False)  #(num_embeddings, embedding_dim)

        # self.embedding = nn.Embedding(n_vocab, Config.lstm_embed, padding_idx=n_vocab - 1)
        self.dropout = nn.Dropout(p=self.dropout)  # dropout训练
        self.lstm = nn.GRU(self.embed_size, hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True)

    def forward(self, input_ids, mask=None):

        out = self.embedding(input_ids)
        # out = self.dropout(emb)
        # print(out,out[:3])
        H, _ = self.lstm(out)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        return H



class TanhAtt(nn.Module):
    def __init__(self, hidden_size):
        super(TanhAtt, self).__init__()
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))

    def forward(self, H, attention_masks):
        M = self.tanh1(H)
        alpha = torch.matmul(M, self.w) #(B,T)
        infinity_tensor = -1e20 * torch.ones_like(alpha)
        alpha_new = torch.where(attention_masks>0,alpha,infinity_tensor)
        alpha = F.softmax(alpha_new, dim=1).unsqueeze(-1)  # [B, T, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]

        return out, alpha


class BiGRUFC(nn.Module):
    def __init__(self, data_name, seed):
        super(BiGRUFC, self).__init__()

        self.model_shortcut = "gru"
        self.att_label_type = None  #cate
        self.data_name = data_name
        self.seed = seed

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.weight_decay = 0.001
        self.warm_up = 0.1
        self.batch_size = 64
        self.lr = 1e-5
        self.patience = 3
        self.epochs = 20
        self.max_len = 80
        self.save_model_path = f"saved_model/{data_name}/{self.model_shortcut}_{str(self.lr)}_{str(seed)}.pt"

        self.hidden_size = 100


        self.gru = BiGRU(data_name, self.hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.hidden_size * 2, Config.NUM_LABELS[data_name])

    def forward(self, x, mask, labels=None, haslabel=True):
        H = self.gru(x, mask)  # [batch_size, seq_len, hidden_size * num_direction]=[32, 50, 256]
        logits = self.fc(H)  # [128, 64]

        if haslabel:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {
                'loss': loss,
                'logits': nn.functional.softmax(logits,dim=1),
                'label': labels,
                "prediction":logits.argmax(1)
                }
        else:
            return {
                'logits': nn.functional.softmax(logits,dim=1),
                "prediction": logits.argmax(1)
            }




class BiGRUAtt(nn.Module):
    def __init__(self, data_name, att_label_type, seed):
        super(BiGRUAtt, self).__init__()

        self.model_shortcut = "gru-att"
        self.att_label_type = "lexicon"  #cate
        self.data_name = data_name
        self.seed = seed
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.weight_decay = 0.001
        self.warm_up = 0.1
        self.batch_size = 64
        self.lr = 1e-5
        self.patience = 3
        self.epochs = 20
        self.max_len = 80
        self.save_model_path = f"saved_model/{data_name}/{self.model_shortcut}_{str(self.lr)}_{str(seed)}_{self.att_label_type}.pt"
        self.lamda = 0.2
        self.hidden_size = 100
        self.gru = BiGRU(data_name, self.hidden_size)
        self.att = TanhAtt(self.hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.hidden_size * 2, Config.NUM_LABELS[data_name])
        self.skew = 2

    @staticmethod
    def normalize_attention(self, att_label):
        att_label = torch.exp(self.skew * att_label)
        att_sum = torch.sum(att_label,dim=1, keepdim=True)
        normalized_att = att_label/att_sum
        return normalized_att


    def forward(self, x, mask, labels=None, haslabel=True):
        att_weight = mask[:,:,0]
        att_weight = self.normalize_attention(att_weight)
        att_label = mask[:,:,1]
        att_label = self.normalize_attention(att_label)
        H = self.gru(x, mask)  # [batch_size, seq_len, hidden_size * num_direction]=[32, 50, 256]
        out, alpha = self.att(H, mask)  # [B, hidden_size * 2]
        out = self.dropout(out)
        logits = self.fc(out)  # [128, 64]

        if haslabel:
            loss_class = torch.nn.functional.cross_entropy(logits, labels)

            loss_attention_flat = torch.nn.functional.mse_loss(input=alpha.reshape(self.batch_size*self.max_len, -1),
                                                               target=att_label.reshape(self.batch_size*self.max_len, -1),
                                                               reduce=False)
            loss_attention_flat = att_weight.reshape(self.batch_size*self.max_len,-1) * loss_attention_flat
            loss_attention_sen = loss_attention_flat.reshape(self.batch_size, -1)
            loss_attention = torch.sum(loss_attention_sen, dim=1)

            loss_sum = loss_class*(1-self.lamda) + loss_attention*(self.lamda)

            return {
                'loss': loss_sum,
                'loss_detail': torch.tensor([loss_class, loss_attention]),
                'logits': nn.functional.softmax(logits,dim=1),
                'label': labels,
                "prediction":logits.argmax(1),
                "alpha": alpha
                }
        else:
            return {
                'logits': nn.functional.softmax(logits,dim=1),
                'label': labels,
                "prediction":logits.argmax(1),
                "alpha":alpha
                }

    @staticmethod
    def compute_metrics(pred,labels):
        return {
            'acc': (labels == pred).sum() / len(labels)
        }