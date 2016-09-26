from operator import mod

import numpy as np
import pandas as pd
from feature_subset_comparison2 import AUCComparator
from infoformulas_listcomp import *
from infoformulas_listcomp import _H
import matplotlib.pyplot as plt
from math import log2

def structure_data(df):
    data = {}
    for f in df:
        mean = np.mean(df[f])
        std = np.std(df[f])
        tmp = df[df[f] == 1]
        cond_mean_y_1 = sum(tmp['target'])/len(tmp[f])
        tmp = df[df[f] == 0]
        cond_mean_y_0 = sum(tmp['target'])/len(tmp[f])
        data[f] = {'mean': mean, 'std': std, 'cond_mean_f_1' : cond_mean_y_1, 'cond_mean_f_0': cond_mean_y_0}
    df_actual = pd.DataFrame(data).transpose()
    """
                                    mean       std
    checking_A11                  0.274  0.446009
    """
    return df_actual

def H_cond(x1_y0, x1_y1, y1):
    y0 = 1-y1
    x0_y0 = 1 - x1_y0
    x0_y1 = 1 - x1_y1
    return y0 * (_H([x1_y0, x0_y0])) + y1 * (_H([x1_y1, x0_y1]))

def inf_gain(h_x, h_cond_x_y):
    return h_x - h_cond_x_y


def get_noise_ig(df_actual, feature, target, N_noise):
    x1 = df_actual.loc[target]['mean']
    igs = list()
    for x1_y1 in np.random.normal(df_actual.loc[feature]['cond_mean_f_1'], df_actual.loc[feature]['std'], N_noise):
        if 0 <= x1_y1 <= 1: # make sure we have a valid probability
            h_cond = H_cond(df_actual.loc[feature]['cond_mean_f_0'], x1_y1, df_actual.loc[feature]['mean'])
            h_x = _H([x1, 1-x1])
            igs.append(inf_gain(h_x, h_cond))
    return igs


def main():
    N_features = 15
    N_noise = 1000
    target = 'target'
    df_actual = pd.read_pickle("datasets/credit/credit_dataset.pickle")
    df_structured = structure_data(df_actual)
    all_features = [f[0] for f in df_structured.iterrows() if f[0] != target]
    print('all features:',all_features)
    # print(df_structured[:1])
    print('--')
    print('Calculating noise for all features... N_noise =',N_noise)
    igs_features = {feature: get_noise_ig(df_structured, feature, target, N_noise) for feature in all_features}
    # {'telephone_A192': [-0.038323405628746032, 0.017350067652810219], 'housing_A153': [0.015780534101964783, 0.010323999315034138], ...}
    print('Noise calculated.')

    print('Calculating auc scores...')
    auc_scores = list()
    for i in range(N_noise): # number of noisy variants considered
        if mod(i,10) == 0:
            print(i)
        information_gains = {f: igs_features[f][i] if len(igs_features[f]) > i else igs_features[f][mod(i,len(igs_features))] for f in all_features} # some features had invalid igs -> they were discarded
        selected_features = sorted(information_gains, key=information_gains.__getitem__, reverse=True)[:N_features] # no of selected features
        # print(selected_features)
        auc_scores.append(np.mean(AUCComparator(df_actual,[],[],target).get_scores(selected_features)))
    print('valid auc scores:',len(auc_scores))
    # print(auc_scores)
    mean = np.mean(auc_scores)
    std = np.std(auc_scores, axis=0)
    plt.plot(auc_scores)
    plt.hold(True)
    plt.plot([mean]*len(auc_scores))
    plt.title('AUC Scores for {} features and {} noisy samples'.format(N_features, N_noise))
    plt.legend(['auc scores. std: {}'.format(std), 'mean: {}'.format(mean)])

    plt.ylabel('AUC')
    fig1 = plt.gcf()
    plt.show()
    plt.hold(False)
    fig1.savefig('noise_auc-{}features-{}samples.png'.format(N_features, N_noise), dpi=100)
    # Todo: use noise for not only x1_y1, but also x1_y0 and y




def test():
    y = [1, 1, 1, 0, 0]
    x = [1, 0, 1, 1, 0]
    t = [1, 1, 0, 0, 0]
    df = pd.DataFrame({'x': x, 'y':y, 'target':t})
    structured = structure_data(df)
    print(structured)
    # test calculations of conditional entropy
    h_cond_f = H_cond(0.5, 2/3, 3/5 )
    h_cond_correct = -1 * (3/5 * (1/3 * log2(1/3) + 2/3 * log2(2/3)) + 2/5 * 2 * (0.5 * log2(0.5)))
    print(h_cond_correct)
    print(h_cond_f)
    assert h_cond_f == h_cond_correct

    h_x = H(x)
    ig_f = inf_gain(h_x, h_cond_f)
    h_x_correct = -(3/5 * log2(3/5) + 2/5 * log2(2/5))
    ig_correct = h_x_correct - h_cond_correct
    assert ig_f == ig_correct



# test()
main()