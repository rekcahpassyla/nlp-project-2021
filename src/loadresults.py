# This script loads the results.hdf5 file that was generated by running
# classify.py in non-training mode (evaluation), and parses the values
# into sensible structures.

import sys
sys.path.append('')

import os
import pandas as pd
import numpy as np

import inputoutput as io



class Record:
    def __init__(self, modelname, trainingset, predictions, labels, data):
        # eg. uk_bert-base-cased
        self.modelname = modelname
        # eg. test_set_uk
        self.trainingset = trainingset
        # these are 3 iterables of the same length
        # and whose indexes match each other
        # eg. the data in inndex 5 has its label in index 5
        # and its prediction in index 5
        # the predictions come from the saved modelname tested on trainingset.
        self.predictions = predictions
        # the data and labels comes from the file name represented by trainingset
        self.labels = labels
        self.data = np.array(data)

    def accuracy(self):
        return (self.predictions == self.labels).mean()

    def misclassified(self, label=None):
        # find all the data items which don't match
        # if label is set to None or any value other than 0 or 1,
        # then will return all mismatches
        # otherwise will return mismatches with the specific label
        unmatched = self.predictions != self.labels

        if label == 1:
            unmatched = unmatched & (self.labels == 1)
        elif label == 0:
            unmatched = unmatched & (self.labels == 0)

        return self.data[unmatched]


def process_results(filename='../results/results.hdf5'):
    # parses the input file and returns structures of processed results and
    # dataframes
    store = pd.HDFStore(filename, 'r')

    # These are pd.DataFrames whose index (rows) are the model and training set
    # eg. 'uk_bert-base-uncased' means that this is the bert-base-uncased model
    # trained on the UK dataset.
    # The DataFrame columns are the test set eg. 'new_test_set' means that
    # this column corresponds to running all the models in the index on
    # new_test_set.json
    losses = store['losses'].unstack()
    accuracies = store['accuracies'].unstack()

    # These are DataFrames whose columns are a multi-index denoting model+training set
    # and test set.
    # These will be further processed for use, so not explaining further
    p = store['predictions']
    l = store['labels']

    # This is a nested dictionary whose first level key is the
    # model-trainingset combination eg. 'uk_bert-base-uncased'
    # and whose second level key is the test set
    results = {}

    for column in p:
        trained_model, test_set = column
        results.setdefault(trained_model, {})
        # load the test set data into a big list
        data, lbls = io.get_data(os.path.join('..', 'datasets', f"{test_set}.json"))
        labels = l[column].dropna()
        assert np.allclose(np.array(lbls), labels.values)
        record = Record(trained_model, test_set,
                        p[column].dropna().values, labels.values,
                        data)
        results[trained_model][test_set] = record
        assert np.allclose(record.accuracy(), accuracies[test_set][trained_model])

    return results, losses, accuracies

