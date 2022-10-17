import transformers
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, BertTokenizer
from transformers import Trainer, TrainingArguments
import torch
from utils.config import Config
import pandas as pd

# import data and process the data into token numbers
data_name = "AMI"
data_fp = Config.DATA_DIC[data_name]["train"]
df = pd.read_csv(data_fp, sep="\t", header=0)
sentences = list(df['cleaned_text'].values.astype('U'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset_id=tokenizer(sentences,max_length=80,truncation=True)['input_ids']

#Use data_collator to create pairs of input data and labels
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Define model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


# Deine training arguments
training_args = TrainingArguments(
    output_dir=f'saved_models/lm/{data_name}',
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
trainer.save_model(f'saved_models/lm/{data_name}')

# model = BertForMaskedLM.from_pretrained(f'saved_model/lm/{data_name}')
# # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# model.cuda()
#
# text = "He is such a [MASK] ."
# tokens = tokenizer.encode(text)
# tokens_tensor = torch.tensor([tokens])
# tokens_tensor = tokens_tensor.to('cuda')
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]
#
# print(predictions.size())  # torch.Size([1, 11, 30522])
#
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