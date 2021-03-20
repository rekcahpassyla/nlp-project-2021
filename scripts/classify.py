from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import sys
sys.path.append('../code')

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
GPU = torch.cuda.is_available()

# If you have a problem with your GPU, set this to "cpu" manually
device = torch.device("cuda:0" if GPU else "cpu")

#device = 'cpu' #torch.device("cuda")

text, labels = io.get_data('../datasets/sarcasm_headlines_dataset.json')


# makes the splitting reproducible - I don't know which one is needed
# so set both
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)

# split train dataset into train, validation and test sets
# Not sure if setting random seed as above is enough to
# make the split reproducible
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

# This string selects the kind of bert model that is used
# All of them have a Huggingface docs page that says what it is
# For example
# https://huggingface.co/bert-base-uncased
# Disorganised list of other models can be found at
# https://huggingface.co/models?search=bert
# I don't know if they can all be downloaded by just changing the string there
bert_type = 'bert-base-uncased'

# I don't know how the tokenizer matches the model type
# but it suggests that we have to use the correct tokenizer for
# the model we want to use
tokenizer = AutoTokenizer.from_pretrained(bert_type)
# Pretrained models come with different heads.
# AutoModelForSequenceClassification consists of Bert + dense layer + softmax
# so we don't need to add any of those, we can just use the output direct
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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

epochs = 20

all_loss = []

train = True


def evalmodel(model, loader):
    model.eval()
    preds = []
    labels = []
    for step, batch in enumerate(loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels += batch['labels'].numpy().tolist()
            this_labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=this_labels)
            this_preds = outputs.logits.argmax(axis=1)
            preds += this_preds.cpu().numpy().tolist()
            loss = outputs[0]
    preds = np.array(preds)
    labels = np.array(labels)
    accuracy = (preds == labels).mean()
    return preds, labels, loss, accuracy

min_filename = "bert_sarcasm_train_min.pt"

if train:

    last_loss = np.inf

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        for step, batch in enumerate(train_loader):

            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))

            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            this_labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=this_labels)
            loss = outputs[0]
            loss.backward()
            l = loss.cpu()
            all_loss.append(l)
            optim.step()
        preds, labels, vloss, accuracy = evalmodel(model, val_loader)
        if vloss < last_loss:
            last_loss = vloss
            torch.save(model.state_dict(), min_filename)

        print(f"Epoch: {epoch} Train loss: {l} "
              f"Val. loss: {vloss.cpu()} Val. acc: {accuracy}")

model.load_state_dict(torch.load(min_filename))

_, _, testloss, test_acc = evalmodel(model, test_loader)
print(f"Test loss: {testloss} Test accuracy: {test_acc}")

torch.cuda.empty_cache()
