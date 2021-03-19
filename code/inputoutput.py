from nltk.tokenize import regexp_tokenize

def parse_json(file_name):
    for line in open(file_name, 'r'):
        yield eval(line)

def tokenize(text):
    return text.split(' ')

def get_data(file_name):
    data = list(parse_json(file_name))
    labels = [x['is_sarcastic'] for x in data]
    headlines = [x['headline'] for x in data]
    # tokenized_headlines = [regexp_tokenize(text) for text in headlines]
    return headlines, labels

x, y = get_data('../datasets/sarcasm_headlines_dataset.json')