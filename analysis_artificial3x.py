from joblib import Parallel, delayed

from analysis_artificial import analysis_general, visualise_results


def do_analysis():
    N_features = [3,5,7,10]#,11,13,16]
    N_samples = 100
    analysis_general("artificial30",N_features, N_samples)
    analysis_general("artificial31",N_features, N_samples)
    analysis_general("artificial32",N_features, N_samples)
    analysis_general("artificial33",N_features, N_samples)
    analysis_general("artificial34",N_features, N_samples)

def evaluate():
    N_features = [3,5,7,10]
    dataset_names = ['artificial30','artificial31','artificial32','artificial33','artificial34']
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features) for dn in dataset_names)
    # N: #data points
    # M: #parameters

# do_analysis()
evaluate()