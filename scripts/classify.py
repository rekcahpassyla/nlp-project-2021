from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast,  BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW

from sklearn.utils.class_weight import compute_class_weight

import inputoutput as io

# specify GPU
device = 'cpu' #torch.device("cuda")

text, labels = io.get_data('../datasets/sarcasm_headlines_dataset.json')


# split train dataset into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(
    text, labels,
    random_state=1,
    test_size=0.3,
    stratify=labels)


val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels,
    random_state=1,
    test_size=0.5,
    stratify=temp_labels)

bert_type = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(bert_type)
model = AutoModelForSequenceClassification.from_pretrained(bert_type)

#train_text = train_text[:1000]
#train_labels = train_labels[:1000]

train_encodings = tokenizer(train_text, truncation=True, padding=True)
val_encodings = tokenizer(val_text, truncation=True, padding=True)
test_encodings = tokenizer(test_text, truncation=True, padding=True)


class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SarcasmDataset(train_encodings, train_labels)
val_dataset = SarcasmDataset(val_encodings, val_labels)
test_dataset = SarcasmDataset(test_encodings, test_labels)

model.to(device)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

epochs = 10

for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    for step, batch in enumerate(train_loader):

        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
