
# coding: utf-8

# In[56]:
import os
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
comparator = AUCComparator(data.copy(), target)
# binarize all data, taking mean as threshold
for f in data:
    b = binarize(data[f], np.mean(data[f]))[0]
    data[f] = b
# print(binarize(data,[1]*len(data)))
# print(data)
data_structured = structure_data(data)
features_ordered = [f for f in data_structured.loc[all_features].sort_values('ig', ascending=False).index]
print(features_ordered)
while n < n_features_end:
    print('number of features: ', n)
    auc_scores = list()
    for i in range(N_samples):
        sys.stdout.write(str(i))
        auc_scores.append(comparator.get_mean_score(features_ordered[:n]))
    # auc_scores = [comparator.get_mean_score(all_features[:n]) for i in range(N_samples)]
    plot_auc_scores(auc_scores, n, N_samples)
    n *= 2
# plt.show()


# In[ ]:



