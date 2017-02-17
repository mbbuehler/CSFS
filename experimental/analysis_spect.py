import numpy as np
from joblib import Parallel, delayed

from CSFSLoader import CSFSLoader
from analysis_std_drop import _conduct_analysis, visualise_results

N_features = [2,3,5,7,11,15,20]
N_samples = 100
dataset_name = 'spect-heart-data'
path = "datasets/spect-heart-data/{}.csv".format(dataset_name)
target = "Diagnosis"

df = CSFSLoader().load_dataset(path)


def do_analysis():

    Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, target, std, N_features, N_samples, dataset_name) for std in np.linspace(0.00001, .3, 500))

def evaluate():
    visualise_results(dataset_name, N_features, show_plot=False, N_samples=N_samples)

def explore():
    print(df[target].describe())

# do_analysis()
# evaluate()
explore()