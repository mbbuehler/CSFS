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

    path = "datasets/artificial/artificial1.pickle"
    df = CSFSLoader().load_dataset(path, format="pickle")
    target = "T"
    dataset_name = "artificial1"
    N_features = [3,5,7]#,11,13,16]
    N_samples = 100

    for std in np.linspace(0.00001, 0.3, 200):
        if start_std < std < end_std:
            sys.stdout.write('std:',std,'\n')
            evaluator = CSFSEvaluator(df, target, fix_std=std)
            best_noisy_selector = CSFSBestUncertainSelector(df, target, fix_std=std)
            for n in N_features:
                aucs = evaluator.evaluate_noisy(n, N_samples, best_noisy_selector)

                filepath = '{}/{}features_{}samples_{:.9f}std'.format(dataset_name, n, len(aucs['best_noisy']), std)
                pickle.dump(aucs, open("pickle-dumps/{}.pickle".format(filepath), 'wb'))

    print('analysis with std ranging from {} to {} is finished.'.format(start_std, end_std))

analysis_general()