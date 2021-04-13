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
        os.path.join('..', 'datasets', 'sarcasm_headlines_dataset.json'))
    print("")
    keep = []
    extract = []
    count = 0
    for i, (headline, label) in enumerate(zip(x, y)):
        if count < 50:
            if int(label) == 1:
                extract.append({'headline': headline, 'is_sarcastic': label})
                count += 1
            else:
                keep.append({'headline': headline, 'is_sarcastic': label})
        else:
            keep.append({'headline': headline, 'is_sarcastic': label})
    with open('extracted_us.json', 'w') as fh:
        for item in extract:
            json.dump(item, fh)
            fh.write("\n")

    with open('keep_us.json', 'w') as fh:
        for item in keep:
            json.dump(item, fh)
            fh.write("\n")
