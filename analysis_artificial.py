import math
import numpy as np
import pickle

import pandas as pd
import sys

from scipy.optimize import curve_fit
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from CSFSLoader import CSFSLoader
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestUncertainSelector
from noise_helper_funcs import structure_data
import os
import re
from CSFSLoader import CSFSLoader
from infoformulas_listcomp import IG

def analysis_general():
    try:
        start_std = float(sys.argv[1])
        end_std = float(sys.argv[2])
        dataset_name = sys.argv[3]
    except:
        print('please use 3 cmd line arguments:')
        print('1. start_std (e.g. 0.00003) and 2. end_std (e.g. 0.0001) 3. Dataset name (e.g. artifical3)')
        exit(-1)

    path = "datasets/artificial/{}.pickle".format(dataset_name)
    df = CSFSLoader().load_dataset(path, format="pickle")
    target = "T"
    N_features = [3,5,7]#,11,13,16]
    N_samples = 100

    for std in np.linspace(0.00001, 0.3, 200):
        if start_std < std < end_std:
            sys.stdout.write('std:{}{}'.format(std,'\n'))
            evaluator = CSFSEvaluator(df, target, fix_std=std)
            best_noisy_selector = CSFSBestUncertainSelector(df, target, fix_std=std)
            for n in N_features:
                aucs = evaluator.evaluate_noisy(n, N_samples, best_noisy_selector)

                filepath = '{}/{}features_{}samples_{:.9f}std'.format(dataset_name, n, len(aucs['best_noisy']), std)
                pickle.dump(aucs, open("pickle-dumps/{}.pickle".format(filepath), 'wb'))

    sys.stdout.write('analysis with std ranging from {} to {} is finished.'.format(start_std, end_std))


def get_result_data(n_features):
    """

    :return: {no_features: {std: auc},...} e.g. {16: {0.200036667: 0.53119531952662713, 0.105176567: 0.57273262130177505
    """
    path = 'pickle-dumps/artificial1'
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
    if n_features:
        results = {r:results[r] for r in n_features}
    return results

def extract_x_y(result, n_features, start_lim=0):
    """
    extracts x and y from results for a certain n_features
    :param result:
    :param n_features:
    :return: x,y where x: std and y:auc
    """
    if n_features not in result.keys():
        print('{} not found'.format(n_features))
        return None
    x = sorted([std for std in result[n_features].keys() if std > start_lim])
    y = [result[n_features][std] for std in x]

    return np.array(x, dtype=float), np.array(y, dtype=float)

def visualize_result():
    dataset_name = "madelon"
    target = 'T'
    N_features = [3,5,7]#,11,13,16]
    N_samples = 100
    results = get_result_data(N_features)
    plt.hold(True)
    fit_curve = False
    start_lim = 0.0
    params = dict()

    def func(x,  w1, p1, w2, p2):
        return w1 * pow(x, p1) + w2 * pow(x, p2)

    # def func(x, w1, p1, w2, p2, w3):
    #     return w1 * pow(x, p1) + w2 * pow(x, p2) + w3 * np.log10(x)

    for n_f in N_features:
        print('== no of features: {}'.format(n_f))
        x,y = extract_x_y(results, n_f, start_lim=0)
        std = np.std(y)
        plt.plot(x, y, alpha=0.5, label='data {} (std={:.3f})'.format(n_f, std))
        if fit_curve:
            x,y = extract_x_y(results, n_f, start_lim=start_lim)
            popt, pcov = curve_fit(func, x, y)
            params[n_f] = popt
            perr = np.sqrt(np.diag(pcov))
            avg_err = np.mean(perr)
            print('params: {} '.format(popt))
            print('errors: {}'.format(perr))
            print('avg error: {}'.format(avg_err))

            plt.plot(x, func(x, *popt), '-k', linewidth=2, label="Fitted {} (avg err: {:.3f})".format(n_f, avg_err))

    plt.legend(loc=1)
    plt.title('auc scores / fitted curves for noisy IG. start fitting at std={}'.format(start_lim))
    plt.xlim([-.01, 0.31])
    plt.xlabel('std')
    plt.ylabel('auc')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('plots/artificial1/std_result.png', dpi=100)

#analysis_general()
visualize_result()