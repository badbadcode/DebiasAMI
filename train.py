from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import torch
import pickle
import os
from sklearn.metrics import accuracy_score,f1_score

from utils.funcs import SetupSeed, getModel, getTrainDevLoader, getTestLoader, getMaskedInput, check_dir
from utils.funcs import load_pkl
from utils.earlystopping import EarlyStopping
from utils.config import Config

def eval(model,dev_dataloader):
    model.eval()  # prep model for evaluation
    dev_labels = []
    dev_losses = []
    dev_logits = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader)):
            b_input_ids = batch[0].to(model.device)
            b_input_mask = batch[1].to(model.device)
            b_labels = batch[2].to(model.device)

            dev_probs_batch, dev_loss_batch = model(input_ids=b_input_ids, input_mask=b_input_mask, labels=b_labels)
            dev_losses.append(dev_loss_batch.mean().item())
            dev_logits.append(dev_probs_batch)  # [tensor(batch_size,NUM_LABELS),tensor(batch_size,NUM_LABELS),.....]
            dev_labels.append(b_labels)


    # calculate average loss over an epoch (all batches)
    dev_loss = np.average(dev_losses)

    dev_logits = [x.tolist() for x in dev_logits]  # [num_batch,batch_size,num_labels]
    dev_labels = [x.tolist() for x in dev_labels]

    dev_logits_flat = sum(dev_logits, [])   # [num_data, num_labels]     sum([[2],[3]],[]) =[2]+[3]==> [2,3] ; sum([[[2]],[[3]]],[])=[[2]] + [[3]]==>[[2], [3]]
    dev_labels_flat = sum(dev_labels, [])

    preds_flat = np.argmax(dev_logits_flat, axis=1)

    dev_acc = accuracy_score(dev_labels_flat, preds_flat)
    dev_f1_hate = f1_score(dev_labels_flat, preds_flat,average="binary")
    dev_f1_weighted = f1_score(dev_labels_flat, preds_flat, average="weighted")


    return dev_loss, dev_acc, dev_f1_hate, dev_f1_weighted


def train(model, train_dataloader):

    optimizer = AdamW(model.parameters(),
                          lr=model.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
                          weight_decay=model.weight_decay)
    total_steps = len(train_dataloader) * model.epochs  # [number of batches] x [number of epochs].
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(model.warm_up * total_steps),
                                                num_training_steps=total_steps)

    early_stopping = EarlyStopping(patience=model.patience, verbose=True,
                                   path=model.save_model_path)  # if valid_loss didn't improve for patience epochs, we stop and save the best one.


    for epoch_i in tqdm(range(0, model.epochs)):

        print("")
        print('================ Epoch {:} / {:} ================='.format(epoch_i + 1, model.epochs))
        print("model_name--", model.model_shortcut, "  train data--", model.data_name, "  lr--", model.lr, "  random_seed--", model.seed)
        print(f'=========={model_shortcut} Training ==========')

        model.train()
        train_losses = []
        for step, batch in enumerate(tqdm(train_dataloader, ncols=80)):
            b_input_ids = batch[0].to(model.device)
            b_input_mask = batch[1].to(model.device)
            b_labels = batch[2].to(model.device)
            model.zero_grad()
            _, loss_batch = model(input_ids=b_input_ids, input_mask=b_input_mask, labels=b_labels)
            # Perform a backward pass to calculate the gradients.
            loss_batch.mean().backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                print(f'\t\t{step // 100}th 100 step loss:', loss_batch.mean())
            train_losses.append(loss_batch.mean().item())

        train_loss = np.average(train_losses)  # 这个epoch的平均训练损失
        print(f'=========={model.model_shortcut} devloping ==========')
        dev_loss,dev_acc,dev_f1_hate, dev_f1_weighted = eval(model, dev_dataloader)
        # _, test_acc = eval(model, test_dataloader)
        # _, unbiased_acc = eval(model, unbiased_dataloader)
        # print("test_acc:",test_acc,"unbiased_acc",unbiased_acc)
        epoch_len = len(str(model.epochs))
        print_msg = (f'[{epoch_i + 1:>{epoch_len}}/{model.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'dev_loss: {dev_loss:.5f}')
        print(print_msg)

        early_stopping(dev_loss, model)
        best_epoch = model.epochs - early_stopping.counter
        if early_stopping.early_stop:
            print(f"Early stopping {model_shortcut}")
            best_epoch = epoch_i + 1 - model.patience
            break
    print("Best epoch is :", best_epoch)


def predictDeltaY(model):
    '''
    有的unbiased测试集，我们没有标签，评估的方式也不是通过标签，而是group之间的差异
    :param model:
    :return:
    '''

    new_input_ids, new_attention_masks, new_labels = getMaskedInput(model, data_name)


    deltaY_sens = []

    with torch.no_grad():
        for step, (id,mask,label) in enumerate(tqdm(zip(new_input_ids, new_attention_masks, new_labels))):

            b_input_ids = torch.tensor(id).to(model.device)
            b_input_mask = torch.tensor(mask).to(model.device)
            b_labels = torch.tensor(label).to(model.device)

            dev_probs2_batch, dev_loss_batch = model(input_ids=b_input_ids, input_mask=b_input_mask, labels=b_labels)
            # print(dev_probs2_batch)
            # print(b_labels)
            dev_probs0_batch = dev_probs2_batch[range(b_labels.size()[0]),1-b_labels]
            # print(dev_probs1_batch)
            dev_probs0_batch = dev_probs0_batch.detach().cpu()
            deltaY_sens.append(dev_probs0_batch.tolist())
    # deltaY_sens = np.asarray(deltaY_sens)

    return deltaY_sens



# 当该module被其它module 引入使用时，其中的"if __name__=="__main__":"所表示的Block不会被执行
if __name__=="__main__":

    data_name = "IMDB-L"
    model_shortcut = "b-ft"
    seed = 42
    SetupSeed(seed)

    model = getModel(model_shortcut, data_name, seed)
    check_dir(model.save_model_path)

    train_dataloader, dev_dataloader = getTrainDevLoader(model, data_name)
    test_dataloader = getTestLoader(model, data_name, "test")
    # unbiased_dataloader = getTestLoader(model, data_name, "unbiased")

    if not os.path.exists(model.save_model_path):
        train(model,train_dataloader)
        print("training is finished")
    else:
        model.load_state_dict(torch.load(model.save_model_path))
        print(f"model is loaded from {model.save_model_path}")

    if model.device == torch.device("cuda"):
        model.cuda()
    _, test_acc, test_f1,test_f1_w = eval(model, test_dataloader)
    print("test_acc", test_acc,"test_f1",test_f1)
    #
    # _, unbiased_acc, unbiased_f1,unbiased_f1_w = eval(model, unbiased_dataloader)
    # print("unbiased_acc", unbiased_acc,"unbiased_f1",unbiased_f1)

    ds = load_pkl(Config.DATA_DIC[data_name])
    ds.deltaY_sens = predictDeltaY(model)
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))
    # np.save(f"data/AMI EVALITA 2018/deltaT_sens.npy", deltaY_sens)