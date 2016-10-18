from random import random

import numpy as np
import pandas as pd
from math import exp

import numpy as np
import pickle
import matplotlib.pyplot as plt

dump_path = 'datasets/artificial/'


def artifical1():
    N_features = 100
    N_samples = 1000

    def get_bin_random_col(mean, std, n, t):
        values = np.random.normal(mean, std, n)
        values = [0 if v < t else 1 for v in values]
        return values

    means = random(N_features)
    stds = random(N_features) / 3
    samples = pd.DataFrame()
    for i in range(N_features):
        t = .5
        samples['X{}'.format(i)] = get_bin_random_col(means[i], stds[i], N_samples, t)
    samples['T'] = get_bin_random_col(0.5, 0.2, N_samples, 0.5)

    print(samples.describe())
    art1_path = dump_path + '/artificial1.pickle'
    pickle.dump(samples, open(art1_path, 'wb'))


def artifical2():
    N_features = 9
    N_samples = 1000

    def get_bin_random_col(mean, std, n, t):
        values = np.random.normal(mean, std, n)
        values = [0 if v < t else 1 for v in values]
        return values

    means = random(N_features)
    stds = random(N_features) / 3
    samples = pd.DataFrame()
    for i in range(N_features):
        t = .5
        samples['X{}'.format(i)] = get_bin_random_col(means[i], stds[i], N_samples, t)
    samples['T'] = get_bin_random_col(0.5, 0.2, N_samples, 0.5)

    print(samples.describe())
    art2_path = dump_path + '/artificial2.pickle'
    pickle.dump(samples, open(art2_path, 'wb'))


def artifical3():
    N_features = 20
    N_samples = 1000

    def get_bin_random_col(mean, std, n, t):
        values = np.random.normal(mean, std, n)
        values = [0 if v < t else 1 for v in values]
        return values

    means = random(N_features)
    stds = random(N_features) / 3
    samples = pd.DataFrame()
    for i in range(N_features):
        t = .5
        samples['X{}'.format(i)] = get_bin_random_col(means[i], stds[i], N_samples, t)
    samples['T'] = get_bin_random_col(0.5, 0.2, N_samples, 0.5)

    print(samples.describe())
    art3_path = dump_path + '/artificial3.pickle'
    pickle.dump(samples, open(art3_path, 'wb'))


def create_artifical(dataset_name, N_features, N_samples, std, relevant_params=[-10, -5, 1, .5, 50]):

    def sigmoid_correct(x, params):
        """

        :param x: list of (binary) features [0, 0, 0, 1, 1]
        :param params: list of params [-1, 2, 10, -5,...]
        :return:
        """
        assert len(x) == len(params)
        alpha = 1
        return alpha / (1 + np.exp(-(sum([params[i] * x[i] for i in range(len(params))]))))  # + w2*x2)))

    def sigmoid_noisy(x, params, std):
        y_actual = sigmoid_correct(x, params)
        y_noisy = np.random.normal(y_actual, std)
        return y_noisy
    def binarize(array_data, threshold=0.5):
        return [1 if a >= threshold else 0 for a in array_data]


    zero_params = [0] * (N_features - len(relevant_params))

    params = relevant_params + zero_params
    np.random.seed()
    random = np.random.random
    X = [binarize([random() for i in range(N_features)]) for j in range(N_samples)]
    y = binarize([sigmoid_correct(X[i], params) for i in range(N_samples)])
    y_noisy = binarize([sigmoid_noisy(X[i], params, std) for i in range(N_samples)])

    file_base_path = 'datasets/artificial/'
    col_names = ['F{}_{}'.format(i, params[i]) for i in range(len(params))]
    col_names.append('T')
    csv_path = file_base_path+dataset_name+'.csv'
    write_csv(csv_path, X, y, col_names)
    readme_path = file_base_path+dataset_name+'.md'
    write_readme(readme_path, X, y, y_noisy, std, len(relevant_params), params)

def write_csv(file_path, X, y, col_names):
    import csv
    file = open(file_path, 'w', newline="")
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    data = [col_names]
    data += [X[i]+[y[i]] for i in range(len(y))]
    writer.writerows(data)
    file.close()

def write_readme(file_path, X, y, y_noisy, std, n_relevant_features, params):
    file = open(file_path, 'w')

    unequal = sum([1 if y[i] != y_noisy[i] else 0 for i in range(len(y))])
    unequal_ratio = unequal/len(y)

    info = (
        ('N_samples', len(X)),
        ('N_features', len(X[0])),
        ('N_relevant features', n_relevant_features),
        ('STD for noise generation', std),
        ('y pos/neg, ratio', "{}/{} {}".format(sum(y), len(y)-sum(y), (len(y)-sum(y))/len(y))),
        ('Noisy y (actual / ratio)', "{} / {}".format(unequal, unequal_ratio)),
        ('Params', ",".join([str(p) for p in params])),
    )
    for a,b in info:
        file.write('{:<30}: {}\n'.format(a,b))
    file.close()

def create_artificial1x():
    create_artifical("artificial10", N_features=10, N_samples=1000, std=0.35)
    create_artifical("artificial11", N_features=20, N_samples=1000, std=0.35)
    create_artifical("artificial12", N_features=50, N_samples=1000, std=0.35)
    create_artifical("artificial13", N_features=100, N_samples=1000, std=0.35)
    create_artifical("artificial14", N_features=200, N_samples=1000, std=0.35)
    create_artifical("artificial15", N_features=1000, N_samples=1000, std=0.35)
    create_artifical("artificial16", N_features=10000, N_samples=1000, std=0.35)

def create_artificial2x():
    create_artifical("artificial20", N_features=20, N_samples=1000, std=0.1)
    create_artifical("artificial21", N_features=20, N_samples=1000, std=0.2)
    create_artifical("artificial22", N_features=20, N_samples=1000, std=0.3)
    create_artifical("artificial23", N_features=20, N_samples=1000, std=0.4)
    create_artifical("artificial24", N_features=20, N_samples=1000, std=0.5)
    create_artifical("artificial25", N_features=20, N_samples=1000, std=0.6)
    create_artifical("artificial26", N_features=20, N_samples=1000, std=0.7)
    create_artifical("artificial27", N_features=20, N_samples=1000, std=0.8)

def create_artificial3x():
    create_artifical("artificial30", N_features=20, N_samples=1000, std=0.35, relevant_params=[-.02, -0.01, 0.01, 0.02, 0.03])
    create_artifical("artificial31", N_features=20, N_samples=1000, std=0.35, relevant_params=[-.3, -.2, -.1, .1, .2])
    create_artifical("artificial32", N_features=20, N_samples=1000, std=0.35, relevant_params=[-2, -1, 1, 2, 3])
    create_artifical("artificial33", N_features=20, N_samples=1000, std=0.35, relevant_params=[-20, -10, 10, 20, 30])
    create_artifical("artificial34", N_features=20, N_samples=1000, std=0.35, relevant_params=[-200, -100, 100, 200, 300])
    create_artifical("artificial35", N_features=20, N_samples=1000, std=0.35, relevant_params=[-200, -100, 1, 2, 0.1])

def create_artificial4x():
    create_artifical("artificial40", N_features=20, N_samples=1000, std=0.35, relevant_params=[-.1, 2, 100])
    create_artifical("artificial41", N_features=20, N_samples=1000, std=0.35, relevant_params=[-.1, 2, 100, 200, -50])
    create_artifical("artificial42", N_features=20, N_samples=1000, std=0.35, relevant_params=[-.1, 2, 100, 200, -50, 20, 5])
    create_artifical("artificial43", N_features=20, N_samples=1000, std=0.35, relevant_params=[-.1, 2, 100, 200, -50, 20, 5, 30, 2, 4, 2, 9])
    create_artifical("artificial44", N_features=20, N_samples=1000, std=0.35, relevant_params=[-.1, 2, 100, 200, -50, 20, 5, 30, 2, 4, 2, 9, 60, 2, -10, 9, 33, -88, 20, 0.4])

def create_artificial5x():
    """
    50 features with 0.5 std, random params between -100 and 100
    Check what influence distribution of T has
    :return:
    """
    for i in range(0,10):
        relevant_params = list(-200 * np.random.random_sample(50) + 100)
        create_artifical("artificial5{}".format(i), N_features=100, N_samples=1000, std=0.5, relevant_params=relevant_params)

def create_artificial6x():
    """
    50 features with 0.5 std, random params between -100 and 100
    Check what influence distribution of T has
    :return:
    """
    for i in range(0,10):
        relevant_params = list(-200 * np.random.random_sample(50) + 100)
        create_artifical("artificial6{}".format(i), N_features=100, N_samples=1000, std=0.2, relevant_params=relevant_params)

def main():
    # create_artificial1x()
    # create_artificial2x()
    # create_artificial3x()
    # create_artificial4x()
    # create_artificial5x()
    # create_artificial6x()


if __name__ == "__main__":
    main()
