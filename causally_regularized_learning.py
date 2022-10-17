import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import pandas as pd
from utils.config import Config
import os

from utils.models import Vector_NN_Classifier, Vector_LogisticNet_Classifier
from utils.funcs import SetupSeed, update_w_one_step, get_sentence_vecs, getGenderIndex
from utils.earlystopping import EarlyStopping

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device:{device}")


def write_add_csv(df, fp):
    dir = "/".join(fp.split("/")[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(fp):
        df.to_csv(fp, index=True, sep=',', header=True)
        print(f"{fp} is saved first time")
    else:
        df.to_csv(fp, mode='a', index=True, sep=',', header=False)
        print(f"{fp} added the new data")


def eval(model, data_loader):
    model.eval()
    m = [0] * 4
    m_items = []
    loss_lst = []  # 算术平均的loss

    for d in data_loader:
        x = torch.stack(d['x'], dim=1).float().to(device)
        x_indexes = d['x_indexes'].to(device)
        labels = d['labels'].to(device)
        results = model(x=x, x_indexes=x_indexes, labels=labels, mode='eval')
        loss = results['loss'].cpu().detach().numpy()
        prediction = results['prediction']
        m_items += (labels * 2 + prediction * 1).tolist()
        loss_lst.append(loss)
    unique, count = np.unique(m_items, return_counts=True)
    data_count = dict(zip(unique, count))
    # print("loss_lst",loss_lst) #[array(54.24918, dtype=float32), array(62.877163, dtype=float32), array(57.798267, dtype=float32), array(57.76215, dtype=float32), array(53.256836, dtype=float32), array(50.448742, dtype=float32), array(59.496258, dtype=float32), array(55.815033, dtype=float32)]

    avg_loss = np.average(loss_lst)
    # print("avg_loss",avg_loss) #56.00573

    for k in data_count.keys():
        m[k] += data_count[k]

    p = m[3] / (m[3] + m[1] + 1)
    r = m[3] / (m[3] + m[2] + 1)
    f1 = 2 * (p * r) / (p + r)
    acc = (m[0] + m[3]) / sum(m)
    _00_number = m[0]
    _01_number = m[1]
    _10_number = m[2]
    _11_number = m[3]

    return {
        'p': p,
        'r': r,
        'f1': f1,
        'acc': acc,
        'detail/_00_number': _00_number,
        'detail/_01_number': _01_number,
        'detail/_10_number': _10_number,
        'detail/_11_number': _11_number,
        'loss': avg_loss
    }


def train(model, data_loader, optimizer):
    model.train()
    loss_item = 0  # 加权平均后的loss
    n_total = 0

    predictions = []
    pred_orign = []
    loss_detial = torch.zeros(3) #最后一个返回的loss_detail,第一项是最后一个batch的加权交叉熵

    for d in data_loader:
        x = torch.stack(d['x'], dim=1).float().to(device)
        x_indexes = d['x_indexes'].to(device)
        labels = d['labels'].to(device)
        results = model(x=x, x_indexes=x_indexes, labels=labels)

        predictions += results['prediction'].tolist()  # argmax后的预测结果
        pred_orign += results['pred_orign'].tolist()  # 预测概率，二维
        loss = results['loss']
        loss_detial = results['loss_detail']  # 这里的loss_detail仍然是最后一个batch的,但是参数的范数（后两个值）跟样本总数也没有关系。

        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_item += loss.item()
        n_total += len(d)

    return loss_item, predictions, pred_orign, (loss_detial).tolist()


def test_report(model, test_dataset, test_fair_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    test_fair_dataloader = DataLoader(test_fair_dataset, batch_size=batch_size)

    results = eval(model, test_dataloader)
    results_fair = eval(model, test_fair_dataloader)

    print("test", results)
    print("test_fair", results_fair)

    return results, results_fair


def start_train_normal(batch_size=64, lr_model=2e-4, epochs=1000, log_name='crl_normal',
                       patience=3, datasets=None, seed=42, model_name="nn"):
    SetupSeed(seed)

    train_dataset, dev_dataset, test_dataset, test_fair_dataset = datasets
    save_model_path = getSaveModelPath(vec_type, seed, log_name, lr_model, model_name)
    writer = SummaryWriter('./logs/' + log_name)

    # print("len(train_dataset['x'][0])", len(train_dataset['x'][0]))

    if model_name == "nn":
        model = Vector_NN_Classifier(hidden_size=len(train_dataset['x'][0]), num_labels=2).to(device)
    elif model_name == "lr":
        model = Vector_LogisticNet_Classifier(hidden_size=len(train_dataset['x'][0]), num_labels=1).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    ###
    # 设置超参
    w = torch.ones(len(train_dataset)).to(device)
    # w = (pow(1/len(train_dataset),0.5) * torch.ones(len(train_dataset))).to(device) #平均权重


    model.set_causal_hyperparameter(w, lambdas)
    #
    ###

    optimizer = AdamW(model.parameters(), lr=lr_model)

    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   path=save_model_path)

    for i in trange(epochs):
        loss1, pred, pred_orign, loss1_detail = train(model, train_dataloader, optimizer)
        print(f"{model_name}-{vec_type}-{log_name} loss1_detail", loss1_detail)
        print("loss1", loss1)
        results = eval(model, dev_dataloader)

        for k in results.keys():
            writer.add_scalar(f'eval/{k}', results[k], global_step=i)
        writer.add_scalar('loss/model', loss1, global_step=i)
        writer.add_scalar('loss/model_examples_loss', loss1_detail[0], global_step=i)
        writer.add_scalar('loss/model_L2_loss', loss1_detail[1], global_step=i)
        writer.add_scalar('loss/model_L1_loss', loss1_detail[2], global_step=i)

        early_stopping(results["loss"], model)
        best_epoch = epochs - early_stopping.counter
        if early_stopping.early_stop:
            print(f"Early stopping ")
            best_epoch = i + 1 - patience
            break

    print(f"the best epoch is {best_epoch}")

    results = {}
    results["model_name"] = [model_name]
    results["log_name"] = [log_name]
    results["vec_type"] = [vec_type]
    results["lr_model"] = [lr_model]
    results["seed"] = [seed]
    results["best_epoch"] = [best_epoch]

    results_test, results_fair = test_report(model, test_dataset, test_fair_dataset)

    for k, v in results_test.items():
        new_k = f"test_{k}"
        results[new_k] = v
    for k, v in results_fair.items():
        new_k = f"fair_{k}"
        results[new_k] = v

    res_df = pd.DataFrame(results, index=[0])
    write_add_csv(res_df, Config.res_path)

    writer.flush()
    writer.close()

    if model_name == "lr" and vec_type != "bert":
        loaded_model = torch.load(save_model_path, map_location=torch.device('cpu'))
        # print(loaded_model.keys())
        # print(loaded_model["linear.weight"].size()) #[1,882]
        sorted_coef = model.TopK(loaded_model, vec_type=vec_type)
        df = pd.DataFrame(sorted_coef, columns=['TOKEN', 'COEF'])
        df.to_csv(save_model_path[:-2] + "csv", header=True, sep="\t")


def start_train_reweight(batch_size=64, lr_model=1e-3, lr_w=1e-4, epochs=1000, log_name='crl_causal',
                         change_distribution=False, patience=3, datasets=None, seed=42, model_name="nn",
                         vec_type="onehot"):
    SetupSeed(seed)
    train_dataset, dev_dataset, test_dataset, test_fair_dataset = datasets

    save_model_path = getSaveModelPath(vec_type, seed, log_name, lr_model, model_name)
    writer = SummaryWriter('./logs/' + log_name)

    ####
    # 设置超参
    Y = torch.tensor(train_dataset['labels']).to(device)
    X = train_dataset['x']

    X = torch.stack([torch.tensor(x) for x in X], dim=0).float().to(device)

    w = torch.randn(len(X)).to(device)  # 随机权重
    # w = (pow(1/len(X),0.5) * torch.ones(len(X))).to(device) #平均权重

    # print("X.size()[1]", X.size()[1]) #(100,882)-onehot

    if model_name == "nn":
        model = Vector_NN_Classifier(hidden_size=X.size()[1], num_labels=2).to(device)
    elif model_name == "lr":
        model = Vector_LogisticNet_Classifier(hidden_size=X.size()[1], num_labels=1).to(device)

    model.set_causal_hyperparameter(w, lambdas)  # 最开始W是随机的正态分布
    I = torch.zeros_like(X).to(device)
    I[X > X.mean(0)] = 1  # 将X高于平均值的地方等于1

    ##  更改X的分布
    if change_distribution:
        X[I == 1] = 1
        X[I == 0] = 0  # 所以X和I是完全相同的？X不转化可不可以（可以，因为这里作者设成了一个参数，说明可以不修改分布，但是处置变量的时候还是要算I），逻辑上是否可以接受？
        train_dataset = train_dataset[:]
        train_dataset['x'] = X
        train_dataset = Dataset.from_dict(train_dataset)

    ###
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    fem_index = getGenderIndex(vec_type, only_female=True)
    # print("fem_index", fem_index)

    optimizer = AdamW(model.parameters(), lr=lr_model)

    early_stopping_val = EarlyStopping(patience=patience, verbose=True,
                                       path=save_model_path)
    early_stopping_JWB = EarlyStopping(patience=patience, verbose=True,
                                       path=save_model_path,delta=5)
    for i in trange(epochs):
        loss1, pred, pred_orign, loss1_detail = train(model, train_dataloader, optimizer)  # 这一步已经对分类模型中的参数做了一个epoch的更新
        pred_orign = torch.tensor(pred_orign).to(device)
        # print("pred_orign",pred_orign.size()) #torch.Size([3200, 2])
        print("\n")
        print(f"{model_name}-{vec_type}-{log_name} loss1_detail", loss1_detail)
        print("loss1", loss1)
        # results = eval(model, dev_dataloader)
        # early_stopping_val(val_loss=results["loss"], model=model)
        # if early_stopping_val.early_stop:
        #     print("Early stopping for dev loss")
        #     best_epoch = i + 1 - patience
        #     print(f"the best epoch is {best_epoch}")
        #     break

    # results_test, results_fair = test_report(model, test_dataset, test_fair_dataset)


    # for j in range(100):

        new_w, loss2, loss2_detail = update_w_one_step(X, Y, model.causal_hyper['w'], I, pred_orign, lambdas[1],
                                                       lambdas[2],
                                                       lambdas[5], lr_w, fem_index=fem_index, device=device)

        model.causal_hyper['w'] = new_w

        print("loss2_detail", loss2_detail)
        print("new_w", new_w)
        print(np.argmax(new_w.detach().cpu().numpy()))

        results = eval(model, dev_dataloader)
        early_stopping_val(val_loss=results["loss"], model=model)
        print("\n")
        early_stopping_JWB(val_loss=loss2 + loss1_detail[1] + loss1_detail[2], model=model)
        # best_epoch = epochs - early_stopping_JWB.counter  # np.min([early_stopping_val.counter, early_stopping_val.counter]) #一直没有停止（不超过patience的loss上升）
        if early_stopping_JWB.early_stop:# or early_stopping_val.early_stop:
            print(f"Early stopping for weight loss in No.{i} training")
            best_epoch = i + 1 - patience
            print(f"the best epoch is {best_epoch} in No.{i} training")
            break

    # for k in results.keys():
    #     writer.add_scalar(f'eval/{k}', results[k], global_step=i)
    # writer.add_scalar('loss/model', loss1, global_step=i)
    # writer.add_scalar('loss/model_examples_loss', loss1_detail[0], global_step=i)
    # writer.add_scalar('loss/model_L2_loss', loss1_detail[1], global_step=i)
    # writer.add_scalar('loss/model_L1_loss', loss1_detail[2], global_step=i)
    #
    # writer.add_scalar('loss/weight', loss2.item(), global_step=i)
    # writer.add_scalar('loss/weight_examples_loss', loss2_detail[0].item(), global_step=i)
    # writer.add_scalar('loss/weight_confounder_loss', loss2_detail[1].item(), global_step=i)
    # writer.add_scalar('loss/weight_L2_loss', loss2_detail[2].item(), global_step=i)
    # writer.add_scalar('loss/weight_avoid_0_loss', loss2_detail[3].item(), global_step=i)


        # elif early_stopping_weight.early_stop and early_stopping_weight.early_stop:
        #     print(f"Early stopping for dev loss")
        #     best_epoch = i + 1 - patience
        #     break

    # print(f"the best epoch is {best_epoch}")
    # writer.add_text('loss/weight', str(model.causal_hyper['w'] ** 2))

    results = {}
    results["model_name"] = model_name
    results["log_name"] = log_name
    results["vec_type"] = vec_type
    results["lr_model"] = lr_model
    results["seed"] = seed
    results["best_epoch"] = best_epoch

    results_test, results_fair = test_report(model, test_dataset, test_fair_dataset)

    for k, v in results_test.items():
        new_k = f"test_{k}"
        results[new_k] = v
    for k, v in results_fair.items():
        new_k = f"fair_{k}"
        results[new_k] = v

    res_df = pd.DataFrame(results, index=[0])
    write_add_csv(res_df, Config.res_path)

    writer.flush()
    writer.close()

    if model_name == "lr" and vec_type != "bert":
        loaded_model = torch.load(save_model_path, map_location=torch.device('cpu'))
        # print(loaded_model.keys())
        # print(loaded_model["linear.weight"].size()) #[1,882]
        sorted_coef = model.TopK(loaded_model, vec_type=vec_type)
        df = pd.DataFrame(sorted_coef, columns=['TOKEN', 'COEF'])
        df.to_csv(save_model_path[:-2] + "csv", header=True, sep="\t")

    # 存储测试结果


# 设置训练和验证数据集的数量,但是我的数据集的标签并不是前后均匀分布的。
def handle_dataset(train_dataset):
    X = train_dataset.to_dict()["x"]
    y = train_dataset.to_dict()["labels"]

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_dataset = Dataset.from_dict({"x": X_train,
                                       "labels": y_train})
    dev_dataset = Dataset.from_dict({"x": X_dev,
                                     "labels": y_dev})

    # df = pd.DataFrame(train_dataset.to_dict())
    #
    # train_1 = df[df["labels"] == 1].sample(n=train_num_pos, random_state=42)
    # train_0 = df[df["labels"] == 0].sample(n=train_num_neg, random_state=42)
    # train_df = pd.concat([train_1, train_0],ignore_index=True)
    #
    # train_dataset = Dataset.from_dict({"x":train_df["x"].tolist(),
    #                                    "labels": train_df["labels"].tolist()})

    return train_dataset, dev_dataset


def getSaveModelPath(vec_type, seed, logname, lr, model_name):
    ckpt_path = f"{Config.CKPT_DIR}/ckpt_{vec_type}/"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_model_path = f'{ckpt_path}checkpoint_{model_name}_{logname}_{str(lr)}_{str(seed)}.pt'
    return save_model_path


###########超参
# # 正例个数
# train_num_pos = 1785
# # 负例个数
# train_num_neg = 4000-1785

# # 测试集中的正负例个数，测试集总数为 2 * eval_num
# eval_num = 1000
# fair_num = 1000
# 一些超参，跟论文公式中对应
lambdas = {
    1: 1,  # confounder loss
    2: 1,  # w l2
    3: 1e-5,  # f l2
    4: 1e-5,  # f l1
    5: 1  # avoid 0
}

# 随机种子
seed = 42
# 批大小
batch_size = 100
# 批数
epochs = 500
# 模型学习率，这里就是神经网络的模型参数
lr = 1e-3
# w学习率
lr_w = 1e-3 # 希望这个先停止，所以学习率设置的稍微大一些。

patience = 5

# 是否干预分布？？将其0/1化？
change_distribution = False

# 梯度裁剪
clip = None
#####超参
# model_name = "lr" # "lr"

if __name__ == '__main__':
    for vec_type in ["onehot", "tfidf", "bert"]:
        # vec_type =
        print(vec_type)

        train_dataset, test_dataset, test_fair_dataset = get_sentence_vecs(vec_type, path=Config.VEC_DIR)

        train_dataset, dev_dataset = handle_dataset(train_dataset)

        train_dataset = train_dataset.add_column('x_indexes', list(range(len(train_dataset))))
        dev_dataset = dev_dataset.add_column('x_indexes', list(range(len(dev_dataset))))
        test_dataset = test_dataset.add_column('x_indexes', list(range(len(test_dataset))))
        test_fair_dataset = test_fair_dataset.add_column('x_indexes', list(range(len(test_fair_dataset))))

        datasets = [train_dataset, dev_dataset, test_dataset, test_fair_dataset]

        for model_name in ["nn"]:
            print(model_name)
            start_train_reweight(epochs=epochs,
                                 lr_model=lr,
                                 lr_w=lr_w,
                                 batch_size=batch_size,
                                 change_distribution=change_distribution,
                                 patience=patience,
                                 datasets=datasets,
                                 seed=seed,
                                 model_name=model_name,
                                 vec_type=vec_type)
            print(f"{vec_type} {model_name} Causal Learning is finished!")

            start_train_normal(epochs=epochs,
                               lr_model=lr, batch_size=batch_size,
                               patience=patience,
                               datasets=datasets,
                               seed=seed,
                               model_name=model_name)
            print(f"{vec_type} {model_name} Normal Learning is finished!")
