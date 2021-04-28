
# analyse profanity values for the various dataset splits
import sys
sys.path.append('../src')

import numpy as np

import pandas as pd
import os
import json
import inputoutput as io

from matplotlib import pyplot as plt

basedir = '../datasets'

files = os.listdir(basedir)

results = {}
stats = {}

for f in files:
    if not f.endswith('_features.json'):
        continue
    fn = os.path.join(basedir, f)
    jsondata = list(io.parse_json(fn))
    tag = f.replace("_features.json", "")
    results[tag] = {}
    results[tag]['sarcastic'] = np.array([
        data['profanity'] for data in jsondata
        if data['is_sarcastic'] == 1
    ])
    results[tag]['non_sarcastic'] = np.array([
        data['profanity'] for data in jsondata
        if data['is_sarcastic'] == 0
    ])
    stats[(tag, 'sarcastic')] = {}
    stats[(tag, 'non_sarcastic')] = {}
    stats[tag, 'sarcastic']['mean'] = results[tag]['sarcastic'].mean()
    stats[tag, 'non_sarcastic']['mean'] = results[tag]['non_sarcastic'].mean()
    stats[tag, 'sarcastic']['std'] = results[tag]['sarcastic'].std()
    stats[tag, 'non_sarcastic']['std'] = results[tag]['non_sarcastic'].std()

stats = pd.DataFrame(stats)

col_order = ['train_set_us', 'train_set_uk', 'train_set_all',
             'test_set_us', 'test_set_uk', 'test_set_all',
             'new_test_set']


test = stats.T.unstack(level=1).stack(level=0).unstack(level=1)
test = test.reindex(col_order)

fig, ax = plt.subplots()
ax.bar(np.arange(0, 7), yerr=test['non_sarcastic']['std'], align='edge', width=0.2, alpha=0.5, ecolor='#1f77b4', capsize=10, height=test['non_sarcastic']['mean'], label='Non sarcastic', error_kw=dict(capthick=0.1))

# First one must be empty. Don't know why. 
labels = ['', 'US/Training', 'UK/Training', 'Combined/Training',
          'US/Test', 'UK/Test', 'Combined/Test',
          'Adversarial']

#ax.set_xticklabels([''] + [str(item) for item in test.index])
ax.set_xticklabels(labels)
plt.xticks(rotation=60)
#plt.margins(0.2)
plt.subplots_adjust(bottom=0.3)
ax.bar(np.arange(0, 7), yerr=test['sarcastic']['std'], align='edge', width=-0.2,  alpha=0.5, ecolor='#ff7f0e', capsize=10, height=test['sarcastic']['mean'], label='Sarcastic', error_kw=dict(capthick=0.1))
plt.legend()
plt.grid()
#plt.title('Mean profanity score for datasets')
plt.show()
plt.savefig('profanity_datasets.png')
