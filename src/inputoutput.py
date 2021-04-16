from nltk.tokenize import regexp_tokenize
import json
import os

def parse_json(file_name):
    for line in open(file_name, 'r'):
        yield json.loads(line.strip())


def tokenize(text):
    return text.split(' ')


def get_data(file_name):
    data = list(parse_json(file_name))
    labels = [x['is_sarcastic'] for x in data]
    headlines = [x['headline'] for x in data]
    # tokenized_headlines = [regexp_tokenize(text) for text in headlines]
    return headlines, labels



if __name__ == '__main__':
    paths = [
        os.path.join('..', 'datasets', 'raw_data', 'sarcasm_headlines_dataset_uk.json'),
        os.path.join('..', 'datasets', 'raw_data', 'sarcasm_headlines_dataset_us.json'),
        os.path.join('..', 'datasets', 'raw_data', 'sarcasm_headlines_dataset.json'),
        os.path.join('..', 'datasets', 'train_set_uk.json'),
        os.path.join('..', 'datasets', 'train_set_us.json'),
        os.path.join('..', 'datasets', 'train_set_all.json'),
    ]
    outdir = os.path.join('..', 'datasets', 'glove')
    for pth in paths:
        x, y = get_data(pth)
        fn = os.path.basename(pth).replace('json', 'txt')
        if fn.startswith('sarcasm_headlines_dataset'):
            fn = fn.replace('sarcasm_headlines_dataset', 'testtrain_set')
        # for each input file, just stick all the headlines together
        # in one big string
        both = []
        pos = []
        neg = []
        for i, (headline, label) in enumerate(zip(x, y)):
            headline = headline.lower()
            tokens = headline.split(" ")
            both += tokens
            if int(label) == 1:
                pos += tokens
            elif int(label) == 0:
                neg += tokens
        with open(os.path.join(outdir, fn), 'w') as fh:
            fh.write(" ".join(both))
        with open(os.path.join(outdir, f"sarcastic_{fn}"), 'w') as fh:
            fh.write(" ".join(pos))
        with open(os.path.join(outdir, f"nonsarcastic_{fn}"), 'w') as fh:
            fh.write(" ".join(neg))
