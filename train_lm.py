import transformers
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, BertTokenizer
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from utils.config import Config
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy


def mask_all(dataset_id, attention_mask):
    new_sens_ids = []
    new_sens_att = []
    for old_sen_id,mask in zip(dataset_id,attention_mask):
        sen_ids = []
        sen_masks = []
        seq_len = sum(mask)
        for i in range(1,seq_len-1):
            new_sen_id = copy.deepcopy(old_sen_id)
            new_sen_id[i] = 103
            sen_ids.append(new_sen_id)
            sen_masks.append(mask)

        new_sens_ids.append(sen_ids)
        new_sens_att.append(sen_masks)

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
    for j,(ids,label) in tqdm(enumerate(zip(new_sens_ids,new_sens_label))):
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

            probs = probs.detach().cpu() #(seq_len-2,seq_len,V)
            deltaT = []
            for i in range(len(label)):
                prob = probs[i][i+1][label[i][i+1]] #[CLS]

                deltaT.append(1-prob.item())
            deltaT_sens.append(deltaT)
        break
    print(deltaT_sens[:2])
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

    train(dataset_id)

    new_sens_ids,_ = mask_all(dataset_id,attention_mask)  # [num_samples, seq_len-2, seq_len]
    new_sens_label = [[x]*(len(x)-2) for x in dataset_id]  # [num_samples, seq_len-2, seq_len] need - [CLS],[SEP]


    deltaT_sens = pred(new_sens_ids,new_sens_label)
    np.save(f"saved_model/lm/AMI/deltaT_sens.npy", deltaT_sens)

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