import os
import pickle
import re
import sys
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from CSFSEvaluator import CSFSEvaluator
from CSFSLoader import CSFSLoader
from CSFSSelector import CSFSBestUncertainSelector

def _conduct_analysis(df, target, std, N_features, N_samples, dataset_name):
    sys.stdout.write('std:{}{}'.format(std,'\n'))
    evaluator = CSFSEvaluator(df, target, fix_std=std)
    best_noisy_selector = CSFSBestUncertainSelector(df, target, fix_std=std)
    for n in N_features:
        aucs = evaluator.evaluate_noisy(n, N_samples, best_noisy_selector)

        filepath = '{}/{}features_{}samples_{:.9f}std'.format(dataset_name, n, len(aucs['best_noisy']), std)
        if not os.path.isdir('pickle-dumps/'+dataset_name):
            os.mkdir('pickle-dumps/'+dataset_name)

        pickle.dump(aucs, open("pickle-dumps/{}.pickle".format(filepath), 'wb'))

def analysis_general(dataset_name, N_features, N_samples):
    path = "datasets/artificial/{}.csv".format(dataset_name)
    df = CSFSLoader().load_dataset(path)
    target = "T"

    Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, target, std, N_features, N_samples, dataset_name) for std in np.linspace(0.00001, 0.3, 500))


if __name__ == "__main__":
    N_features = [3,5,7,10]#,11,13,16]
    N_samples = 100

    analysis_general("artificial10",N_features, N_samples)
    analysis_general("artificial11",N_features, N_samples)
    analysis_general("artificial12",N_features, N_samples)
    analysis_general("artificial13",N_features, N_samples)
    analysis_general("artificial14",N_features, N_samples)