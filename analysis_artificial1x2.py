
from joblib import Parallel, delayed

from analysis_artificial import analysis_general, visualise_results


N_features = [3,5,7,10]
dataset_base = ['artificial10_2','artificial11_2','artificial12_2','artificial13_2','artificial14_2']
dataset_names = [dn+'_true' for dn in dataset_base] + [dn+'_noisy' for dn in dataset_base]

def do_analysis():
    N_samples = 1
    for dn in dataset_names:
        analysis_general(dn, N_features, N_samples)

def evaluate():
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features, start_lim=0.035) for dn in dataset_names)
    # N: #data points
    # M: #parameters

do_analysis()
# evaluate()