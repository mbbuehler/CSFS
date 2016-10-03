import numpy as np
import pandas as pd
from numpy.random import random
import numpy as np
from sklearn.preprocessing import binarize

N_features = 100
N_samples = 1000


N_features = 3
N_samples = 10

means = random(N_features)
stds = random(N_features)/3
samples = pd.DataFrame()
for i in range(N_features):
     values = np.random.normal(means[i], stds[i], N_samples)
     samples['X{}'.format(i)] = values


print(samples)
print(binarize(samples, .5))
exit()
print(means)
print(std)

means_binned = np.histogram(means, 10)
print(means_binned)

mean = 0.7
std = 0.07
std = 0.000001
x1 = np.random.normal(mean, std, N_samples)
print(x1)
binned = np.histogram(x1,5)[0]
print(binned)
dummies = pd.get_dummies(binned)
print(dummies)


# for i in range(N_features):
#     np.random.normal(mean, std)