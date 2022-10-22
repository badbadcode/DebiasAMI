import numpy as np
from utils.config import Config
from transformers import BertTokenizer
import pandas as pd



if __name__=='__main__':
    data_name = "AMI"
    delta_T = np.load("data/AMI EVALITA 2018/deltaT_sens.npy", allow_pickle=True)
    delta_Y = np.load("data/AMI EVALITA 2018/deltaY_sens.npy", allow_pickle=True)

    att_weight = list(delta_T)
    delta_Y = list(delta_Y)

    att_label = []
    for ts,ys in zip(att_weight,delta_Y):
        labels = []
        for t,y in zip(ts,ys):
            labels.append(y/t)
        att_label.append(labels)


    data_fp = Config.DATA_DIC[data_name]["train"]
    df = pd.read_csv(data_fp, sep="\t", header=0)

    sentences = list(df['cleaned_text'].values.astype('U'))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    for ids, labels, weights in zip(dataset_id[22:],att_label[22:],att_weight[22:]):
        imp_word = tokenizer.convert_ids_to_tokens(ids[1+np.argmax(labels)])
        # print(np)
        print("the most important word", imp_word,np.max(labels),weights[np.argmax(labels)])
        for id, lbl, wei in zip(ids[1:-1], labels, weights):
            token = tokenizer.convert_ids_to_tokens(id)
            print(token,lbl,wei)
        break