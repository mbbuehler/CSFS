import numpy as np
import pandas as pd
from numpy.random import random
import numpy as np
from sklearn.preprocessing import binarize
import pickle

dump_path = 'datasets/artificial/'

def artifical1():
     N_features = 100
     N_samples = 1000

     def get_bin_random_col(mean, std, n, t):
          values = np.random.normal(mean, std, n)
          values = [0 if v < t else 1 for v in values]
          return values

     means = random(N_features)
     stds = random(N_features)/3
     samples = pd.DataFrame()
     for i in range(N_features):
          t = .5
          samples['X{}'.format(i)] = get_bin_random_col(means[i], stds[i], N_samples, t)
     samples['T'] = get_bin_random_col(0.5, 0.2, N_samples, 0.5)

     print(samples.describe())
     art1_path = dump_path+'/artificial1.pickle'
     pickle.dump(samples, open(art1_path, 'wb'))

def artifical2():
     N_features = 5
     N_samples = 1000

     def get_bin_random_col(mean, std, n, t):
          values = np.random.normal(mean, std, n)
          values = [0 if v < t else 1 for v in values]
          return values

     means = random(N_features)
     stds = random(N_features)/3
     samples = pd.DataFrame()
     for i in range(N_features):
          t = .5
          samples['X{}'.format(i)] = get_bin_random_col(means[i], stds[i], N_samples, t)
     samples['T'] = get_bin_random_col(0.5, 0.2, N_samples, 0.5)

     print(samples.describe())
     art2_path = dump_path+'/artificial2.pickle'
     pickle.dump(samples, open(art2_path, 'wb'))

def artifical3():
     N_features = 20
     N_samples = 1000

     def get_bin_random_col(mean, std, n, t):
          values = np.random.normal(mean, std, n)
          values = [0 if v < t else 1 for v in values]
          return values

     means = random(N_features)
     stds = random(N_features)/3
     samples = pd.DataFrame()
     for i in range(N_features):
          t = .5
          samples['X{}'.format(i)] = get_bin_random_col(means[i], stds[i], N_samples, t)
     samples['T'] = get_bin_random_col(0.5, 0.2, N_samples, 0.5)

     print(samples.describe())
     art3_path = dump_path+'/artificial3.pickle'
     pickle.dump(samples, open(art3_path, 'wb'))
artifical3()