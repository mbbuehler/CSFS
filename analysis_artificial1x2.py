import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from CSFSSelector import CSFSBestActualSelector
from analysis_artificial import analysis_general, visualise_results


N_features = [3,5,7,10]
dataset_base = ['artificial10_2','artificial11_2','artificial12_2','artificial13_2','artificial14_2']
dataset_names = [dn+'_true' for dn in dataset_base] + [dn+'_noisy' for dn in dataset_base]
# dataset_names = ['artificial10_2_true']

def do_analysis():
    N_samples = 100
    for dn in dataset_names:
        analysis_general(dn, N_features, N_samples)

def evaluate():
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features) for dn in dataset_names)
    # N: #data points
    # M: #parameters

def explore_dataset():
    df = pd.read_csv('datasets/artificial/artificial12_2_true.csv')
    target='T'
    # print(df.describe())
    best_actual_selector = CSFSBestActualSelector(df, target)

    dict_ig = best_actual_selector.dict_ig
    best_sorted = sorted(dict_ig, key=dict_ig.__getitem__, reverse=True)
    [print("{} {}".format(f, dict_ig[f])) for f in best_sorted[:10]]

def explore_dumps():
    dn = 'artificial12_2_true'
    n_feat = 5
    std = 0.000010000
    data = pickle.load(open('pickle-dumps/{}/{}features_100samples_{:.9f}std.pickle'.format(dn,n_feat,std),'rb'))
    print(data)
    def most_selected_features(data, n):
        return data['best_noisy_features_count'].most_common(n)
    print('most selected features (noisy selection):',most_selected_features(data, 10))
    print('mean noisy:', np.mean(data['best_noisy']))
    print('mean best:', np.mean(data['best']))


# do_analysis()
# evaluate()
explore_dumps()
explore_dataset()