# annotate each dataset with features
# order is important and must be preserved


import sys
sys.path.append('../src')

import os
import json
import inputoutput as io

from profanity_check import predict_prob

basedir = '../datasets'

files = os.listdir(basedir)

for fn in files:
    if not fn.endswith('.json') or fn.endswith('_features.json'):
        continue
    tag, _ = fn.split(".")
    featurefn = os.path.join(basedir, f'{tag}_features.json')
    headlines, labels = io.get_data(os.path.join(basedir, fn))
    with open(featurefn, 'w') as fh:
        for headline, label in zip(headlines, labels):
            # replace f**k and c**t with their actual words before running prob
            headline_orig = headline.replace("f**k", "fuck")
            headline_orig = headline_orig.replace("c**t", "cunt")
            prob = predict_prob([headline_orig])
            out = dict(headline=headline, is_sarcastic=label, profanity=str(prob[0]))
            json.dump(out, fh)
            fh.write("\n")


