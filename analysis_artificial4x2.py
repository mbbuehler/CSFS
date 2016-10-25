from joblib import Parallel, delayed

from analysis_artificial import analysis_general, visualise_results, create_dn_names_from_base


dataset_base = ['artificial40_2','artificial41_2','artificial42_2','artificial43_2','artificial44_2']
dataset_names = create_dn_names_from_base(dataset_base)
N_features = [3,5,7,10]

def do_analysis():
    N_samples = 100
    for dn in dataset_names:
        analysis_general(dn, N_features, N_samples)

def evaluate():
    Parallel(n_jobs=4)(delayed(visualise_results)(dn, N_features, start_lim=0) for dn in dataset_names)
    # N: #data points
    # M: #parameters

do_analysis()
# evaluate()