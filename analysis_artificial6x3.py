
from joblib import Parallel, delayed

from analysis_noisy_means_drop import analysis_general, visualise_results
from util.util_features import create_dn_names_from_base

N_features = [1, 5, 10, 25, 50, 75]
dataset_base = ['artificial6-{}_3'.format(i) for i in N_features]
dataset_names = create_dn_names_from_base(dataset_base)
N_samples = 1000
target = 'T'

def do_analysis():
    for dn in dataset_names:
        analysis_general(dn, N_features, N_samples, target)

def evaluate():
    N_features = [1, 5, 25, 75]
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features, N_samples=N_samples, target=target) for dn in dataset_names)
    # N: #data points
    # M: #parameters

# do_analysis()
evaluate()