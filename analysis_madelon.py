import math
import numpy as np
import pickle

import pandas as pd
import sys
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from CSFSLoader import CSFSLoader
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestUncertainSelector
from noise_helper_funcs import structure_data
import os
import re
"""
Example starting command:
python3 analysis_madelon.py 0.000026416 0.00005
"""



def std_analysis():
    path = 'pickle-dumps/madelon/std_samples'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

    results = dict()
    pattern = r'(\d+)features_100samples_(.*?)std'
    for f in files:
        match = re.match(pattern, f)
        # print(f)
        no_features = int(match.group(1))
        std = float(match.group(2))

        if no_features not in results.keys():
            results[no_features] = dict()

        results[no_features][std] = (np.mean(pickle.load(open(os.path.join(path,f), 'rb'))['best_noisy']))

    legend_str = []
    for no_f in sorted(results):
        sorted_keys = sorted(results[no_f])
        plt.plot(sorted_keys, [results[no_f][key] for key in sorted_keys])
        aucs_only = list(results[no_f].values())
        print(aucs_only)
        legend_str.append('{} (std: {:.5f} *10^13)'.format(no_f, np.std(aucs_only)*math.pow(10,13)))
    # plt.legend([no_f for no_f in sorted(results)])
    plt.legend(legend_str, loc=4)
    plt.title('auc scores for different #features with noisy IG')
    plt.xlabel('std')
    plt.ylabel('auc')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('plots/madelon/std_result.png', dpi=100)

# def analysis2_fix():
#     try:
#         start_std = float(sys.argv[1])
#         end_std = float(sys.argv[2])
#     except:
#         print('please use 2 cmd line arguments:')
#         print('1. start_std (e.g. 0.00003) and 2. end_std (e.g. 0.0001)')
#         exit(-1)
#
#     path = 'datasets/madelon/madelon_combined.csv'
#     dataset_name = "madelon"
#     target = 'target'
#     N_features = [2,3,5,7,11,13,16]
#     N_samples = 100
#
#     df = CSFSLoader.load_dataset(path, format='csv')
#     df = preprocess(df)
#     df = df[['F1','F2','F3','F4','F5','F6','target']]
#     print(df.describe())
#
#     for std in np.linspace(0.001, 0.5, 1000):
#         if start_std < std < end_std:
#             print('std', std)
#             evaluator = CSFSEvaluator(df, target, fix_std=std)
#             best_noisy_selector = CSFSBestUncertainSelector(df, target, fix_std=std)
#             n = 3
#             aucs = evaluator.evaluate_noisy(n, N_samples, best_noisy_selector)
#
#     print('analysis with std ranging from {} to {} is finished.'.format(start_std, end_std))


def analysis2():
    """
    Trying to fit an exponential curve to auc scores (with varying std)
    :return:
    """
    try:
        start_std = float(sys.argv[1])
        end_std = float(sys.argv[2])
    except:
        print('please use 2 cmd line arguments:')
        print('1. start_std (e.g. 0.00003) and 2. end_std (e.g. 0.0001)')
        exit(-1)

    path = 'datasets/madelon/madelon_combined.csv'
    dataset_name = "madelon"
    target = 'target'
    N_features = [2,3,5,7,11,13,16]
    N_samples = 100

    df = CSFSLoader.load_dataset(path, format='csv')
    df = preprocess(df)

    # start_std = 0.000026416

    for std in np.linspace(0.00011, 0.3, 1000):
        if start_std < std < end_std:
            print('std', std)
            evaluator = CSFSEvaluator(df, target, fix_std=std)
            best_noisy_selector = CSFSBestUncertainSelector(df, target, fix_std=std)
            for n in N_features:
                aucs = evaluator.evaluate_noisy(n, N_samples, best_noisy_selector)

                filepath = '{}/std_samples/{}features_{}samples_{:.9f}std'.format(dataset_name, n, len(aucs['best_noisy']), std)
                pickle.dump(aucs, open("pickle-dumps/{}.pickle".format(filepath), 'wb'))

    print('analysis with std ranging from {} to {} is finished.'.format(start_std, end_std))


def analysis1():
    path = 'datasets/madelon/madelon_combined.csv'
    dataset_name = "madelon"
    target = 'target'
    N_features = [2,3,5,7]#,11,13,16]
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
    # std_analysis()
    # analysis2_fix()