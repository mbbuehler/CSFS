from joblib import Parallel, delayed

from analysis_artificial import analysis_general, visualise_results


dataset_names = ['artificial30','artificial31','artificial32','artificial33','artificial34','artificial35']

def do_analysis():
    N_features = [3,5,7,10]#,11,13,16]
    N_samples = 100
    for dn in dataset_names:
        analysis_general(dn, N_features, N_samples)

def evaluate():
    N_features = [3,5,7,10]
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features, start_lim=0) for dn in dataset_names)
    # N: #data points
    # M: #parameters

do_analysis()
evaluate()