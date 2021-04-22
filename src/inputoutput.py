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
    basepath = os.path.join('..', 'datasets')
    files = [
        'new_test_set',
        'test_set_uk',
        'test_set_all',
        'train_set_uk',
        'train_set_all',
    ]
    outdir = basepath

    for fn in files:
        pth = os.path.join(basepath, f"{fn}.json")
        out = os.path.join(basepath, f"{fn}2.json")
        print(f"Processing {pth}")
        x, y = get_data(pth)

        with open(out, 'w') as fh:
            for headline, category in zip(x, y):
                item = {
                    "headline": headline.lower(),
                    "is_sarcastic": category
                }
                json.dump(item, fh)
                fh.write("\n")

