#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import sys

from transformers.modeling_outputs import SequenceClassifierOutput

sys.path.append("../src")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, AutoConfig, BertTokenizerFast,  BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW

from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

import inputoutput as io

#%%
class SarcasmFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, features=None):
        self.encodings = encodings   # Bert encodings
        self.labels = labels
        # features must be a list of features
        self.features = features

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["features"] = torch.tensor(self.features[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BertPlus(nn.Module):
    def __init__(self, bert_type):
        super(BertPlus, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_type)
        self.bertcfg = AutoConfig.from_pretrained(bert_type)
        self.dropout = nn.Dropout(self.bertcfg.hidden_dropout_prob)
        # This layer will have 1 extra input which is the feature (there's only one at the moment)
        self.combiner = nn.Linear(self.bertcfg.hidden_size + 1, 256)
        self.classifier = nn.Linear(256, 2)

    def forward(self, ids, attention_mask, labels, features):
        outputs = self.bert(ids, attention_mask=attention_mask),
        outputs = outputs[0]
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        next_in = torch.hstack([torch.unsqueeze(features, dim=0).T, pooled_output])
        next_out = self.combiner(next_in)
        logits = self.classifier(next_out)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
            this_features = batch["features"].to(device)
            outputs = model.forward(
                input_ids, attention_mask=attention_mask,
                labels=this_labels, features=this_features)
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

    text, labels, features = io.get_data("../datasets/%s" % args_pack["train_set"], features=True)
    test_text, test_labels, test_features = io.get_data("../datasets/%s" % args_pack["test_set"],
                                                        features=True)

    # makes the splitting reproducible - I don"t know which one is needed
    # so set both
    seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    # split train dataset into train and validation
    # train_test_split is deterministic if the random_state is set as below
    train_text, val_text, train_labels, val_labels, train_features, val_features = train_test_split(
        text, labels, features,
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
    #model = AutoModelForSequenceClassification.from_pretrained(bert_type)
    model = BertPlus(bert_type)

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    val_encodings = tokenizer(val_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)

    min_filename = "%s_%s_features.pt" % (args_pack["name"], bert_type)
    if results_dir is not None and not train:
        min_filename = os.path.join(results_dir, min_filename)
    train_dataset = SarcasmFeaturesDataset(train_encodings, train_labels, train_features)
    val_dataset = SarcasmFeaturesDataset(val_encodings, val_labels, val_features)
    test_dataset = SarcasmFeaturesDataset(test_encodings, test_labels, val_features)

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
                    print("{}:{}  Batch {:>5,}  of  {:>5,}.".format(
                        bert_type, args_pack["train_set"], step, len(train_loader)))

                optim.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                this_labels = batch["labels"].to(device)
                this_features = batch["features"].to(device)
                outputs = model.forward(
                    input_ids, attention_mask=attention_mask,
                    labels=this_labels, features=this_features)
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
    print(f"{args_pack['test_set']}: Test loss: {testloss} Test accuracy: {test_acc}")

    torch.cuda.empty_cache()
    return actual, expected, testloss, test_acc

#%%

if __name__ == "__main__":
    # specify GPU
    GPU = torch.cuda.is_available()

    # If you have a problem with your GPU, set this to "cpu" manually
    device = torch.device("cuda:0" if GPU else "cpu")

    device = "cpu"

    TRAIN = True

    model_names = [#"bert-base-uncased",
                   "distilbert-base-uncased"
                   ]
    names = ["uk",
             #"us",
             #"all"
             ]
    train_sets = ["train_set_uk_features.json",
                  #"train_set_us_features.json",
                  # "train_set_all_features.json"
                  ]
    test_sets = ["test_set_uk_features.json",
                 #"test_set_us_features.json",
                 # "test_set_all_features.json"
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
    results_file = pd.HDFStore('results_features.hdf5', 'w')
    results_file['predictions'] = pd.DataFrame(predictions)
    results_file['labels'] = pd.DataFrame(labels)
    results_file['losses'] = pd.Series(losses)
    results_file['accuracies'] = pd.Series(accuracies)
    results_file.close()
