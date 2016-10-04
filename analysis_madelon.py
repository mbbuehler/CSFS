import numpy as np
import pickle

import pandas as pd
from sklearn.preprocessing import binarize

from CSFSLoader import CSFSLoader
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestUncertainSelector
from noise_helper_funcs import structure_data


def analysis2():
    """
    Trying to fit an exponential curve to auc scores (with varying std)
    :return:
    """
    path = 'datasets/madelon/madelon_combined.csv'
    dataset_name = "madelon"
    target = 'target'
    N_features = [2,3,5,7,11,13,16]
    N_samples = 100

    df = CSFSLoader.load_dataset(path, format='csv')
    df = preprocess(df)

    for std in np.linspace(0.00001, 0.00011, 1000):
        print('std', std)
        evaluator = CSFSEvaluator(df, target, fix_std=std)
        best_noisy_selector = CSFSBestUncertainSelector(df, target, fix_std=std)
        for n in N_features:
            aucs = evaluator.evaluate_noisy(n, N_samples, best_noisy_selector)

            filepath = '{}/{}features_{}samples_{}std'.format(dataset_name, n, len(aucs['best_noisy']), std)
            pickle.dump(aucs, open("pickle-dumps/{:.9}.pickle".format(filepath), 'wb'))


def analysis1():
    path = 'datasets/madelon/madelon_combined.csv'
    dataset_name = "madelon"
    target = 'target'
    N_features = [2,3,5,7,11,13,16]
    N_samples = 100

    df = CSFSLoader.load_dataset(path, format='csv')
    df = preprocess(df)

    std = .05

    evaluator = CSFSEvaluator(df, target, fix_std=std)

    for n in N_features:
        aucs = evaluator.evaluate(n, N_samples)
        evaluator.plot(aucs, {'dataset': dataset_name, 'N_features': n, 'N_samples': N_samples, 'std': std})

def preprocess(data):
    for f in data:
        b = binarize(data[f], np.mean(data[f]))[0]
        data[f] = b
    # data_structured = structure_data(data[:50])
    return data

if __name__ == "__main__":
    analysis2()