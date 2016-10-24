
from joblib import Parallel, delayed

from analysis_artificial import analysis_general, visualise_results


N_features = [3,5,7,10]
dataset_base = ['artificial20_2','artificial21_2','artificial22_2','artificial23_2','artificial124_2','artificial125_2','artificial126_2']
dataset_names = [dn+'_true' for dn in dataset_base] + [dn+'_noisy' for dn in dataset_base]
# dataset_names = ['artificial10_2_true']

def do_analysis():
    N_samples = 100
    for dn in dataset_names:
        analysis_general(dn, N_features, N_samples)

def evaluate():
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features) for dn in dataset_names)
    # N: #data points
    # M: #parameters

# do_analysis()
evaluate()