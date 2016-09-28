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