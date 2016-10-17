import numpy as np
from joblib import Parallel, delayed

from CSFSLoader import CSFSLoader
from analysis_std_drop import _conduct_analysis, visualise_results

N_features = [2,3,5,7,11,15,20]
N_samples = 100
dataset_name = 'spect-heart-data'

def do_analysis():
    path = "datasets/spect-heart-data/{}.csv".format(dataset_name)
    df = CSFSLoader().load_dataset(path)
    target = "Diagnosis"

    Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, target, std, N_features, N_samples, dataset_name) for std in np.linspace(0.00001, 0.3, 4))

def evaluate():
    visualise_results(dataset_name, N_features, show_plot=False, N_samples=100)

do_analysis()
evaluate()