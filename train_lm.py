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