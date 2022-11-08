import transformers
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, BertTokenizer
from transformers import Trainer, TrainingArguments

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
import os
import pickle


from utils.funcs import load_pkl, mask_syn
from utils.config import Config
from utils.data_class import Counterfactual


def organize_data(ds):
    '''

    :param df:
    :return:
        dataset_id: the input of the training with size [num_samples, seq_len]
        mask_sens_ids : [num_samples, num_token_in_vocab, seq_len]
        mask_sens_label:[num_samples, num_token_in_vocab, seq_len]
        index_vocab_sens: [num_samples, num_token_in_vocab]
        prob_ids_sens: [num_samples, num_token_in_vocab, num_syns_token]
    '''
    df = ds.train
    vocab_df = ds.antonym_vocab
    sentences = list(df['text'].values.astype('U'))

    encoded_inputs = tokenizer(sentences, max_length=80, truncation=True)
    dataset_id = encoded_inputs['input_ids'] #label [num_samples, seq_len]
    attention_mask = encoded_inputs['attention_mask'] #label [num_samples, seq_len]

    mask_sens_ids, _, index_vocab_sens, prob_ids_sens = mask_syn(dataset_id,attention_mask, tokenizer, vocab_df)
    '''
    dataset_id: [num_samples, seq_len]
    mask_sens_ids : [num_samples, num_token_in_vocab, seq_len]
    '''
    mask_sens_label = [[old_id]*len(mask_ids_lst) for mask_ids_lst, old_id in zip(mask_sens_ids,dataset_id)]
    # print(len(mask_sens_ids), mask_sens_ids[0])  # 1707, [num_token_in_vocab, seq_len]
    return dataset_id, mask_sens_ids, mask_sens_label, index_vocab_sens,prob_ids_sens


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


def pred(new_sens_ids,new_sens_label, index_vocab_sens, prob_ids_sens):
    '''
    :param new_sens_ids: [num_samples, num_token_in_vocab, seq_len]
    :param new_sens_label: [num_samples, num_token_in_vocab, seq_len]
    :param index_vocab_sens: [num_samples, num_token_in_vocab]
    :param prob_ids_sens: [num_samples, num_token_in_vocab, num_syns_token]
    :return:
    '''
    model = BertForMaskedLM.from_pretrained(model_path)
    if model.device == torch.device("cuda"):
        model.cuda()
    deltaT_sens = []
    for j,(ids,label,index_vocab, prob_ids) in enumerate(tzip(new_sens_ids,
                                                                  new_sens_label,
                                                                  index_vocab_sens,
                                                                  prob_ids_sens)):
        # (num_token_in_vocab, seq_len)
        tokens_tensor = torch.tensor(ids)
        tokens_tensor = tokens_tensor.to(model.device)
        if tokens_tensor.size()[0] == 0:
            deltaT_sens.append([])
        else:
            with torch.no_grad():
                outputs = model(tokens_tensor)#, output_hidden_states=True)
                # embedding = outputs.hidden_states[0]
                # print(embedding.size())   # (seq_len-2, seq_len, 768)
                # print(type(outputs), len(outputs))  # <class 'transformers.modeling_outputs.MaskedLMOutput'> 2
                # print(len(outputs.hidden_states)) #13
                predictions = outputs.logits
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
                # # print(ids)

                probs = probs.detach().cpu().numpy() #(num_token_in_vocab,seq_len,V)

                deltaT = []

                # probs_ids: [num_token_in_vocab, num_syns_token]
                for i in range(len(index_vocab)):  # num_token_in_vocab
                    token_pos = index_vocab[i]
                    p_sum = 0
                    for prob_id_tok in prob_ids[i]:
                        p_sum = p_sum + probs[i][token_pos][prob_id_tok]
                    deltaT.append(1-p_sum)

                # for check_index in range(len(index_vocab)):
                #     print("\nthe first mask sample of the first sample-->",tokenizer.convert_ids_to_tokens(ids[check_index]))
                #     token_pos = index_vocab[check_index]
                #     print("the index of the mask word:", token_pos)
                #     print("the masked word is:",tokenizer.convert_ids_to_tokens(label[check_index][token_pos]))
                #     top_k_values, top_k_indices = torch.topk(probs[check_index][token_pos], 5, sorted=True)
                #     print("tok_k model predicted words:", tokenizer.convert_ids_to_tokens(top_k_indices))
                #     print("top_k value:",top_k_values)
                #     sysn_wors = tokenizer.convert_ids_to_tokens(prob_ids[check_index])
                #     print("the sysn of this word is",sysn_wors)
                #
                #     print("the sysn's prob is", [probs[check_index][token_pos][sy] for sy in prob_ids[check_index]])
                #
                #     print("model predicts:", tokenizer.convert_ids_to_tokens(torch.argmax(probs[check_index][token_pos],dim=-1).item()))
                #     print("the prob of the real masked word :", probs[check_index][token_pos][label[check_index][token_pos]])
                #     print("the 1-p_sum",deltaT[check_index])

                deltaT_sens.append(deltaT)
                # if j == 3:
                #     break

    # print(deltaT_sens)
    return deltaT_sens


if __name__ == '__main__':
    # import data and process the data into token numbers
    for data_name in ["IMDB-S","IMDB-L","KINDLE"]:
        model_path = f'saved_model/lm/{data_name}'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        ds = load_pkl(Config.DATA_DIC[data_name])
        # df = ds.train
        dataset_id, mask_sens_ids, mask_sens_label, index_vocab_sens, prob_ids_sens = organize_data(ds)

        if not os.path.exists(model_path):
            train(dataset_id)
            print("training is finished")
        else:
            print(f"to load lm model from {model_path}")

        deltaT_sens = pred(mask_sens_ids, mask_sens_label, index_vocab_sens, prob_ids_sens)
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