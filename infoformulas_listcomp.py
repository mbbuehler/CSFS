from math import log

import math
# Checked with results from http://www.cs.man.ac.uk/~pococka4/MIToolbox.html (7.9.16)
import pandas as pd


def joint_probabilities(X, Y):
    """
    :param x:  x = [1, 0, 1, 1, 0];
    :param y:  y = [1, 1, 1, 0, 0]
    :return: joint probabilities
    """
    n_min = min([len(X), len(Y)])
    probs = dict()

    for x in set(X):
        for y in set(Y):
            probs[(x,y)] = sum([1 for k in range(n_min) if X[k] == x and Y[k] == y])/n_min
    return probs

def _H(probs):
    """
    :param probs: probabilities [0.4, 0.1, 0.5]
    :return:
    """
    return sum([-p_0 * log(p_0,2) for p_0 in probs if p_0 != 0])

def H(x):
    """
    :param X: X = [1, 1, 1, 0, 0]
    :return: float
    """
    n = len(x)
    # calculate probabilities
    probs = [sum([1 for e in x if e == xi])/n for xi in set(x)]
    entropy = _H(probs)
    return entropy

def prob(x, xi):
    """
    calculcates probability for a realization xi in x
    :param x: vector
    :param xi: element in x
    :return: float f : f>=0 and f<=1
    """
    return sum([1 for x0 in x if x0 == xi])/len(x)

def IG_from_series(instance, h_x, identifier='p'):
    """

    :param instance: series
    :param h_x: H(x) entropy of target variable
    :return:
    """

    cond_h = H_X_Y_from_series(instance, identifier=identifier)
    return h_x - cond_h

def H_X_Y_from_series(instance, identifier):
    """
    Calculates conditional entropy from p and cond_p
    ONLY FOR BINARY
    :param x: list, e.g. [ 0.3, 0.7]
    :param y_cond: conditional probability list, e.g. [0.1, 0.9]
    :return:
    """
    # print('H f=0:',_H([instance['p|f=0'], 1-instance['p|f=0']]))
    # print('H f=1:',_H([instance['p|f=1'], 1-instance['p|f=1']]))
    # print((1-instance['p']) * _H([instance['p|f=0'], 1-instance['p|f=0']]))
    #
    # print((instance['p']) * _H([instance['p|f=1'], 1-instance['p|f=1']]))
    normal = identifier
    cond_0 = '{}|f=0'.format(identifier)
    cond_1 = '{}|f=1'.format(identifier)
    return (1-instance[normal]) * _H([instance[cond_0], 1-instance[cond_0]]) \
           + (instance[normal]) * _H([instance[cond_1], 1-instance[cond_1]])


def H_X_Y(x, y):
    """
    Calculates conditional entropy without using conditional probability
    :param x:  x = [1, 0, 1, 1, 0];
    :param y:  y = [1, 1, 1, 0, 0]
    :return: Conditional Entropy
    """
    joint_p = joint_probabilities(x, y)
    entropy = sum([joint_p[(xi,yi)] * log(prob(y, yi) / joint_p[(xi, yi)], 2) if joint_p[(xi, yi)] != 0 else 0 for yi in set(y) for xi in set(x)])
    return entropy

def MI(x, y):
    """
    :param x:  x = [1, 0, 1, 1, 0];
    :param y:  y = [1, 1, 1, 0, 0]
    :return: Mutual Information
    """
    joint_p = joint_probabilities(x, y)
    mi = sum([joint_p[(xi,yi)] * log(joint_p[(xi,yi)] / (prob(x, xi) * prob(y, yi)), 2) if prob(x, xi) * prob(y, yi) != 0 and joint_p[(xi, yi)] != 0 else 0 for yi in set(y) for xi in set(x)])
    return mi

def IG(x,y):
    """
    :param x:  x = [1, 0, 1, 1, 0];
    :param y:  y = [1, 1, 1, 0, 0]
    :return: Information Gain
    """
    return H(x) - H_X_Y(x, y)


def IG_fast(x, y, h_x):
    return h_x - H_X_Y(x,y)

def H_XY(x, y):
    """
    :param x:  x = [1, 0, 1, 1, 0];
    :param y:  y = [1, 1, 1, 0, 0]
    :return: Joint Entropy
    """
    joint_p = joint_probabilities(x, y)
    joint_e = sum([-joint_p[(xi,yi)] * log(joint_p[(xi,yi)],2) if joint_p[(xi,yi)] != 0 else 0 for yi in set(y) for xi in set(x)])
    return joint_e

def SU(x,y):
    """
    Symmetrical uncertainty (Yu and Liu 2004)
    :param x:
    :param y:
    :return:
    """
    return 2*(IG(x,y)/(H(x) + H(y)))

def cond_means(X,Y,y):
    """
    You can call _H(cond_means(X,Y,1) to get entropy on conditional means
    :param X: array of discrete variables, e.g. [1, 1, 1, 0, 0]
    :param Y: array of discrete variables, e.g. [1, 0, 1, 1, 0]
    :param y: realisation of var Y, e.g. 1
    :return: the probabilities for all realisations x of X
    """
    X_selected = [X[i] for i in range(len(Y)) if Y[i] == y]
    ratios = [prob(X_selected, xi) for xi in set(X_selected)]
    return ratios

def H_cond(x1_y0, x1_y1, y1):
    """
    with (cond.) probabilities
    :param x1_y0:
    :param x1_y1: (noisy) cond p
    :param y1:
    :return:
    """
    y0 = 1-y1
    x0_y0 = 1 - x1_y0
    x0_y1 = 1 - x1_y1
    return y0 * (_H([x1_y0, x0_y0])) + y1 * (_H([x1_y1, x0_y1]))

def test():
    # print(' == Start Tests ==')
    y = [1, 1, 1, 0, 0]
    x = [1, 0, 1, 1, 0]
    #
    entropy_x = H(x)
    print('entropy %e' % entropy_x)
    assert round(entropy_x,7) == .9709506
    #
    cond_e = H_X_Y(x, y)
    print('cond_e %e' % cond_e)
    assert round(cond_e,7) == .9509775
    joint_e = H_XY(x, y)
    print('joint_e %e' % joint_e)
    assert round(joint_e,6) == 1.921928
    mi = MI(x, y)
    print('mi %e' % mi)
    assert(round(mi,8)) == .01997309
    ig = IG(x, y)
    print('ig %e' % ig)
    assert(round(ig,8)) == .01997309
    # IG == MI
    assert(round(ig,8) == round(mi,8))
    su = SU(x,y)
    print('su %e' % su)
    assert(round(su,8)) == .02057066
    cond_p = cond_means(x,y,1)
    print('cond_p (y=1)', cond_p)
    assert(cond_p == [0.3333333333333333, 0.6666666666666666])
    e_cond_p = _H(cond_p)
    print('H on cond_p (y=1) %e' % e_cond_p)
    assert(e_cond_p == 0.9182958340544896)
    X = [1,2,3,4,5,4,5,6]
    Y = [1,2,3,3,3,3,3,3]
    cond_p = cond_means(X,Y,3)
    assert(cond_p == [0.16666666666666666, 0.3333333333333333, 0.3333333333333333, 0.16666666666666666])
    assert(_H(cond_p) == 1.9182958340544893)
    cond_p = cond_means(x,y,0)
    print('cond_p (y=0)', cond_p)
    print(_H(cond_p))


    print(H(x) + H(y) - H_XY(x,y))
    print(H(x) - H_X_Y(x,y))
    print(H(y) - H_X_Y(y,x))
    print(MI(x,y))
    print(IG(x,y))


    instance = pd.Series({'p': 3/5, 'p|f=0': 0.5, 'p|f=1': 2/3})
    cond_entropy = H_X_Y_from_series(instance)
    cond_entropy_true = .9509775
    print(cond_entropy,' == ', cond_entropy_true, '?')
    assert round(cond_entropy, 5) == round(cond_entropy_true, 5)


    h_x = _H([3/5, 1-3/5])
    ig_true = h_x - cond_entropy_true
    # print(instance)
    ig = IG_from_series(instance, h_x)
    print(ig, ' == ', ig_true, '?')
    assert round(ig, 5) == round(ig_true, 5)

    print(_H([5/14, 5/14, 4/14]))

    print(' == Tests OK ==')

if __name__ == '__main__':
    test()


