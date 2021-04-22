#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import sys
sys.path.append("../src")

import os
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

from tqdm import tqdm

import inputoutput as io

#%%
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#%%
def evalmodel(model, loader):
    model.eval()
    preds = []
    labels = []
    for step, batch in enumerate(loader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels += batch["labels"].numpy().tolist()
            this_labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=this_labels)
            this_preds = outputs.logits.argmax(axis=1)
            preds += this_preds.cpu().numpy().tolist()
            loss = outputs[0]
    preds = np.array(preds)
    labels = np.array(labels)
    accuracy = (preds == labels).mean()
    return preds, labels, loss, accuracy



def bulk_eval(
        args_pack,
        epochs = 50, train = True, results_dir = None):

    text, labels = io.get_data("../datasets/%s" % args_pack["train_set"])
    test_text, test_labels = io.get_data("../datasets/%s" % args_pack["test_set"])

    # makes the splitting reproducible - I don"t know which one is needed
    # so set both
    seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    # split train dataset into train and validation
    # train_test_split is deterministic if the random_state is set as below
    train_text, val_text, train_labels, val_labels = train_test_split(
        text, labels,
        random_state=1,
        test_size=0.3,
        stratify=labels)

    # This string selects the kind of bert model that is used
    # All of them have a Huggingface docs page that says what it is
    # For example
    # https://huggingface.co/bert-base-uncased
    # Disorganised list of other models can be found at
    # https://huggingface.co/models?search=bert
    # I don"t know if they can all be downloaded by just changing the string there
    bert_type = args_pack["model_name"]

    # I don"t know how the tokenizer matches the model type
    # but it suggests that we have to use the correct tokenizer for
    # the model we want to use
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    # Pretrained models come with different heads.
    # AutoModelForSequenceClassification consists of Bert + dense layer + softmax
    # so we don"t need to add any of those, we can just use the output direct
    model = AutoModelForSequenceClassification.from_pretrained(bert_type)

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    val_encodings = tokenizer(val_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)

    min_filename = "%s_%s.pt" % (args_pack["name"], bert_type)
    if results_dir is not None:
        min_filename = os.path.join(results_dir, min_filename)
    train_dataset = SarcasmDataset(train_encodings, train_labels)
    val_dataset = SarcasmDataset(val_encodings, val_labels)
    test_dataset = SarcasmDataset(test_encodings, test_labels)

    model.to(device)

    # we do not care about shuffling the order of the test data.
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    optim = AdamW(model.parameters(), lr=5e-5)

    epochs = epochs

    all_loss = []

    if train:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        last_loss = np.inf
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for step, batch in enumerate(train_loader):

                if step % 50 == 0 and not step == 0:
                    print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_loader)))

                optim.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                this_labels = batch["labels"].to(device)
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
    kws = {}
    if device == 'cpu':
        kws['map_location'] = device

    model.load_state_dict(torch.load(min_filename, **kws))

    actual, expected, testloss, test_acc = evalmodel(model, test_loader)
    if not train:
        print(f"Evaluating model {bert_type} from {min_filename} with test set {args_pack['test_set']}")
    print(f"Test loss: {testloss} Test accuracy: {test_acc}")

    torch.cuda.empty_cache()
    return actual, expected, testloss, test_acc

#%%

if __name__ == "__main__":
    # specify GPU
    GPU = torch.cuda.is_available()

    # If you have a problem with your GPU, set this to "cpu" manually
    device = torch.device("cuda:0" if GPU else "cpu")

    device = "cpu"

    TRAIN = False

    model_names = ["bert-base-uncased",
                   "distilbert-base-uncased"]
    names = ["uk", "us", "all"
             ]
    train_sets = ["train_set_uk.json",
                  "train_set_us.json", "train_set_all.json"
                  ]
    test_sets = ["test_set_uk.json", "test_set_us.json", "test_set_all.json"
                 ]

    args_packs = []
    for model_name in model_names:
        # standard models
        for name, train_set, test_set in zip(names, train_sets, test_sets):
            args_packs.append({
                "name" : name, "train_set" : train_set,
                "test_set" : test_set, "model_name" : model_name
            })
        if not TRAIN:
            # cross testing: do all the combinations.
            # only need the name and the test set.
            cases = [
                ("us", "test_set_uk.json"),
                ("us", "test_set_all.json"),
                ("us", "new_test_set.json"),
                ("uk", "test_set_us.json"),
                ("uk", "test_set_all.json"),
                ("uk", "new_test_set.json"),
                ("all", "test_set_uk.json"),
                ("all", "test_set_us.json"),
                ("all", "new_test_set.json"),
            ]
            for name, test_set in cases:
                train_set = f"test_set_{name}.json"
                args_packs.append({
                    "name" : name, "train_set" : train_set,
                    "test_set" : test_set, "model_name" : model_name
                })
    # first level: type of model and what it was trained on
    # second level: name of test set
    predictions = {}
    labels = {}
    losses = {}
    accuracies = {}
    for args_pack in args_packs:
        tag = f"{args_pack['name']}_{args_pack['model_name']}"

        actual, expected, testloss, test_acc = bulk_eval(
            args_pack = args_pack,
            epochs = 50,
            train = TRAIN,
            # where to read the saved results from
            # if we are not training
            results_dir = '../results2')
        test_set_tag = args_pack['test_set'].replace('.json', '')
        # the index of these Series is not meaningful, it is just the
        # index number of the data item.
        # Both are matched
        predictions[(tag, test_set_tag)] = pd.Series(actual)
        labels[(tag, test_set_tag)] = pd.Series(expected)
        losses[(tag, test_set_tag)] = float(testloss.cpu().numpy())
        accuracies[(tag, test_set_tag)] = test_acc

    results_file = pd.HDFStore('results.hdf5', 'w')
    results_file['predictions'] = pd.DataFrame(predictions)
    results_file['labels'] = pd.DataFrame(labels)
    results_file['losses'] = pd.Series(losses)
    results_file['accuracies'] = pd.Series(accuracies)
    results_file.close()
