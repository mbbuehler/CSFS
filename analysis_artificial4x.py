
from joblib import Parallel, delayed

from analysis_artificial import analysis_general, visualise_results


def do_analysis():
    N_features = [3,5,7,10]#,11,13,16]
    N_samples = 100
    analysis_general("artificial40",N_features, N_samples)
    analysis_general("artificial41",N_features, N_samples)
    analysis_general("artificial42",N_features, N_samples)
    analysis_general("artificial43",N_features, N_samples)
    analysis_general("artificial44",N_features, N_samples)

def evaluate():
    N_features = [3,5,7,10]
    dataset_names = ['artificial40','artificial41','artificial42','artificial43','artificial44']
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features, start_lim=0.035) for dn in dataset_names)
    # N: #data points
    # M: #parameters

# do_analysis()
evaluate()