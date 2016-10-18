import csv
import re
from infoformulas_listcomp import IG
import numpy as np
from joblib import Parallel, delayed
from CSFSLoader import CSFSLoader
from analysis_std_drop import _conduct_analysis, visualise_results

N_features = [3,5,7,11,15,20]
N_samples = 100
dataset_name = 'Olympia_2_update'

def do_analysis():
    path = "datasets/olympia/{}.csv".format(dataset_name)
    df = CSFSLoader().load_dataset(path)
    target = "medals"

    Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, target, std, N_features, N_samples, dataset_name) for std in np.linspace(0.00001, .3, 500))

def evaluate():
    visualise_results(dataset_name, N_features, show_plot=False, N_samples=N_samples)


def explore():
    path = "datasets/olympia/Olympia_2_update.csv"
    df = CSFSLoader().load_dataset(path)
    target = "medals"

    print(df.describe())
    df = df[:20]
    import pandas as pd
    ig_data = {f:IG(df[target], df[f]) for f in df}

    ordered = sorted(ig_data, key=ig_data.__getitem__, reverse=True)
    for f in ordered:
        print(f, ig_data[f])

    ordered.remove(target)
    ordered.remove('id')
    print('best 10:')
    print(sorted(ordered[:10]))
    print('worst 10:')
    print(sorted(ordered[-10:]))

    print(ig_data['population growth rate_(1.541, 2.254]'])

    mean_05_f = [f for f in df if 0.3 < np.mean(df[f]) < 0.7]
    mean_05 = df[mean_05_f]
    print(mean_05.describe())
    for f in mean_05_f:
        print(f, np.mean(df[f]))



def extract_prefix():
    file_name = "datasets/olympia/featuresOlympia_2_update.csv"
    reader = csv.reader(open(file_name,'r'), delimiter=",", quotechar='"')

    features = [(re.sub(r'_[\[\(\d].*?$','', row[0])) for row in reader]
    for f in set(features):
        print(f)

do_analysis()
# evaluate()

# visualize_result()
# explore()
# extract_prefix()
