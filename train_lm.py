import transformers
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, BertTokenizer
from transformers import Trainer, TrainingArguments

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle


from utils.funcs import load_pkl, mask_all
from utils.config import Config
from utils.data_class import Counterfactual


def organize_data(df):

    sentences = list(df['text'].values.astype('U'))

    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    attention_mask = encoded_inputs['attention_mask'] #label [num_samples, seq_len]

    mask_sens_ids,_ = mask_all(dataset_id,attention_mask)  # [num_samples, seq_len-2, seq_len]
    mask_sens_label = [[x]*(len(x)-2) for x in dataset_id]  # [num_samples, seq_len-2, seq_len] need - [CLS],[SEP]

    return dataset_id, mask_sens_ids, mask_sens_label


def train(dataset_id):
    #Use data_collator to create pairs of input data and labels
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # Define model
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Deine training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=32,
        save_steps=50,  # for save the models
        save_total_limit=2,
        gradient_accumulation_steps=64,
        prediction_loss_only=True,
        logging_steps=50,  # for printing loss
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_id,
    )
    trainer.train()
    trainer.save_model(model_path)

def pred(new_sens_ids,new_sens_label):
    # pred:
    model = BertForMaskedLM.from_pretrained(model_path)
    model.cuda()
    deltaT_sens = []
    for j,(ids,label) in enumerate(tqdm(zip(new_sens_ids,new_sens_label))):
        # (seq_len-2,seq_len)
        tokens_tensor = torch.tensor(ids)
        tokens_tensor = tokens_tensor.to('cuda')
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            # print(predictions.size()) # #(seq_len-2,seq_len,V)
            probs = nn.functional.softmax(predictions, dim=2)
            # print("the first mask sample of the first sample")
            # print(tokenizer.convert_ids_to_tokens(ids[0]))
            # print("the masked word is:",tokenizer.convert_ids_to_tokens(label[0][1]))
            # top_k_values, top_k_indices = torch.topk(probs[0][1], 5, sorted=True)
            # print("tok_k model predicted words:", tokenizer.convert_ids_to_tokens(top_k_indices))
            # print("top_k value:",top_k_values)
            # print("model predicts:", tokenizer.convert_ids_to_tokens(torch.argmax(probs[0][1],dim=-1).item()))
            # print("the prob of the real masked word :", probs[0][1][label[0][1]])

            probs = probs.detach().cpu().numpy() #(seq_len-2,seq_len,V)
            deltaT = []
            for i in range(len(label)):
                prob = probs[i][i+1][label[i][i+1]] #[CLS]
                deltaT.append(1-prob)
            deltaT_sens.append(deltaT)

    print(deltaT_sens[:2])
    return deltaT_sens


if __name__ == '__main__':
    # import data and process the data into token numbers
    data_name = "IMDB-L"
    model_path = f'saved_model/lm/{data_name}'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ds = load_pkl(Config.DATA_DIC[data_name])
    df = ds.train
    dataset_id, mask_sens_ids, mask_sens_label = organize_data(df)

    if not os.path.exists(model_path):
        train(dataset_id)
        print("training is finished")
    else:
        print(f"to load lm model from {model_path}")

    deltaT_sens = pred(mask_sens_ids, mask_sens_label)

    print(deltaT_sens)

    ds.deltaT_sens = deltaT_sens
    pickle.dump(ds, open(Config.DATA_DIC[data_name], "wb"))



# print(predictions.size())  # torch.Size([1, 11, 30522]) will add [cls]/[sep] automatically
#
# predicted_index = torch.argmax(predictions[0, 6]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token) # "a"
# print(torch.max(predictions[0,6]).item())

# top_k = 5
# #probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
# top_k_values, top_k_indices = torch.topk(predictions[0, 5], top_k, sorted=True)
#
# for i, pred_idx in enumerate(top_k_indices):
#     pred_idx = pred_idx.item()
#     predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
#     token_weight = top_k_values[i].item()
#     print(predicted_token)
#     print(token_weight)