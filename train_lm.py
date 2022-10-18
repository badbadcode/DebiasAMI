import transformers
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, BertTokenizer
from transformers import Trainer, TrainingArguments
import torch
from utils.config import Config
import pandas as pd
import numpy as np

def mask_i(ids,i):
    new_ids = []
    for j in range(len(ids)):
        if j !=i:
            new_ids.append(ids[j])
        else:
            new_ids.append(103)
    return new_ids

def mask_all(dataset_id, attention_mask):
    new_sens_ids = []
    new_sens_att = []
    for old_sen_id,mask in zip(dataset_id,attention_mask):
        sen_ids = []
        sen_masks = []
        for i in range(len(old_sen_id)):
            if mask[i] == 1:  # this position is not padding
                new_id_i = mask_i(old_sen_id, i)
                sen_ids.append(new_id_i)
                sen_masks.append(mask)
        new_sens_ids.append(sen_ids[1:-1])
        new_sens_att.append(sen_masks[1:-1])
    return new_sens_ids, new_sens_att


def train(dataset_id):
    #Use data_collator to create pairs of input data and labels
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # Define model
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Deine training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=10,
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
    for i,ids,label in enumerate(zip(new_sens_ids,new_sens_label)):
        # (seq_len-2,seq_len)
        tokens_tensor = torch.tensor(ids)
        tokens_tensor = tokens_tensor.to('cuda')
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0] #(seq_len-2,seq_len,V)
            deltaT = []
            for i in range(len(label)):
                prob = predictions[i][i+1][label[i][i+1]] #[CLS]
                deltaT.append(1-prob)
            deltaT_sens.append(deltaT)
    return deltaT_sens

if __name__ == '__main__':
    # import data and process the data into token numbers
    data_name = "AMI"

    model_path = f'saved_model/lm/{data_name}'

    data_fp = Config.DATA_DIC[data_name]["train"]
    df = pd.read_csv(data_fp, sep="\t", header=0)
    sentences = list(df['cleaned_text'].values.astype('U'))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    attention_mask = encoded_inputs['attention_mask'] #label [num_samples, seq_len]

    new_sens_ids,_ = mask_all(dataset_id,attention_mask)  # [num_samples, seq_len-2, seq_len]
    new_sens_label = [[x]*(len(x)-2) for x in dataset_id]  # [num_samples, seq_len-2, seq_len] need - [CLS],[SEP]

    train(dataset_id)
    deltaT_sens = pred(new_sens_ids,new_sens_label)

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