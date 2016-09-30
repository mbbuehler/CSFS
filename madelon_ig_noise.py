
# coding: utf-8

# In[56]:
import os
from operator import mod
from random import shuffle

import sys

from sklearn.preprocessing import binarize

from feature_subset_comparison2 import AUCComparator
import matplotlib.pyplot as plt
from math import log2
import numpy as np
import pandas as pd


# In[57]:
from infoformulas_listcomp import IG
from noise_helper_funcs import structure_data

data = pd.read_csv('datasets/madelon/madelon_combined.csv')
data.describe()


# In[58]:

data[:3]


# In[59]:

target = 'target'
N_samples = 100
all_features = [f for f in data if f != target]
# all_features = all_features[:200]
all_features[:3]


# In[60]:

def plot_auc_scores(auc_scores, N_features, N_samples):
    mean = np.mean(auc_scores)
    std = np.std(auc_scores)
    plt.plot(auc_scores)
    plt.hold(True)
    plt.plot([mean]*len(auc_scores))
    plt.title('Madelon Dataset. AUC Scores for {} features with highest IG'.format(N_features))
    plt.legend(['auc scores. std: {}'.format(std), 'mean: {}'.format(mean)], loc=4)
    plt.xlabel('# samples (total: {})'.format(N_samples))
    plt.ylabel('AUC')
    fig1 = plt.gcf()
    plt.draw()
    plt.hold(False)
    fig1.savefig('plots/madelon/IG_ACTUAL_auc-{}features_{}samples.png'.format(N_features, N_samples), dpi=100)
    print('done N_features:',N_features)


# In[62]:

n_features_start = 2
n_features_end = len(all_features)
n = n_features_start
N_noise = 10
comparator = AUCComparator(data.copy(), target)

# binarize all data, taking mean as threshold
for f in data:
    b = binarize(data[f], np.mean(data[f]))[0]
    data[f] = b
# print(binarize(data,[1]*len(data)))
# print(data)
data_structured = structure_data(data[:50])
# features_ordered = [f for f in data_structured.loc[all_features].sort_values('ig', ascending=False).index]
# print(features_ordered)
while n < n_features_end/2:
    print('number of features: ', n)
    from noise_analysis import get_noise_ig
    igs_features = {feature: get_noise_ig(data_structured, feature, target, N_noise) for feature in all_features}

    print('Noise calculated.')

    print('Calculating auc scores...')
    auc_scores = list()
    for i in range(N_noise): # number of noisy variants considered
        print('fast',comparator.fast)
        if mod(i,10) == 0:
            print(i)
        information_gains = {f: igs_features[f][i] if len(igs_features[f]) > i else igs_features[f][mod(i,len(igs_features))] for f in all_features} # some features had invalid igs -> they were discarded
        selected_features = sorted(information_gains, key=information_gains.__getitem__, reverse=True)[:n] # no of selected features
        # print(selected_features)
        auc_scores.append(np.mean(comparator.get_mean_score(selected_features)))
    print('valid auc scores:',len(auc_scores))
    plot_auc_scores(auc_scores, n, N_samples)


    n *= 2





