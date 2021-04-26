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

#for fn in files:
for fn in ['new_test_set.json']:
    if not fn.endswith('.json') or fn.endswith('_features.json'):
        continue
    tag, _ = fn.split(".")
    featurefn = os.path.join(basedir, f'{tag}_features.json')
    headlines, labels = io.get_data(os.path.join(basedir, fn))
    with open(featurefn, 'w') as fh:
        for headline, label in zip(headlines, labels):
            prob = predict_prob([headline])
            out = dict(headline=headline, is_sarcastic=label, profanity=str(prob[0]))
            json.dump(out, fh)
            fh.write("\n")


