
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

stats.T
