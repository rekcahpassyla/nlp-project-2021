from nltk.tokenize import regexp_tokenize
import json
import os

def parse_json(file_name):
    for line in open(file_name, 'r'):
        yield json.loads(line.strip())
        #yield eval(line)


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
        os.path.join('..', 'datasets', 'sarcasm_headlines_dataset.json'))
    print("")
