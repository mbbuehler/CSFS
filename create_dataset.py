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


def create_artifical(dataset_name, N_features, N_samples, std):

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


    relevant_params = [-10, -5, 1, .5, 50]
    zero_params = [0] * (N_features - len(relevant_params))

    params = relevant_params + zero_params
    np.random.seed()
    random = np.random.random
    X = [binarize([random() for i in range(N_features)]) for j in range(N_samples)]
    y = binarize([sigmoid_correct(X[i], params) for i in range(N_samples)])
    y_noisy = binarize([sigmoid_noisy(X[i], params, std) for i in range(N_samples)])
    # print(y)
    # print(y_noisy)


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
    print(data)
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
        ('Noisy y (actual / ratio)', "{} / {}".format(unequal, unequal_ratio)),
        ('Params', ",".join([str(p) for p in params])),
    )
    for a,b in info:
        file.write('{:<30}: {}\n'.format(a,b))
    file.close()

def main():
    create_artifical("artificial10", N_features=10, N_samples=1000, std=0.35)
    create_artifical("artificial11", N_features=20, N_samples=1000, std=0.35)
    create_artifical("artificial12", N_features=50, N_samples=1000, std=0.35)
    create_artifical("artificial13", N_features=100, N_samples=1000, std=0.35)
    create_artifical("artificial14", N_features=200, N_samples=1000, std=0.35)


if __name__ == "__main__":
    main()
