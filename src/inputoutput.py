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
    x, y = get_data(
        os.path.join('..', 'datasets', 'sarcasm_headlines_dataset_uk.json'))
    print("")
    keep = []
    extract = []
    pos = 0
    neg = 0
    N = 1500
    for i, (headline, label) in enumerate(zip(x, y)):
        if int(label) == 1:
            if pos < N:
                extract.append({'headline': headline, 'is_sarcastic': label})
                pos += 1
        elif int(label) == 0:
            if neg < N:
                extract.append({'headline': headline, 'is_sarcastic': label})
                neg += 1
        if pos == N and neg == N:
            keep.append({'headline': headline, 'is_sarcastic': label})
    with open('test_set_uk.json', 'w') as fh:
        for item in extract:
            json.dump(item, fh)
            fh.write("\n")

    with open('train_set_uk.json', 'w') as fh:
        for item in keep:
            json.dump(item, fh)
            fh.write("\n")
