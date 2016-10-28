import csv
import re

import pandas as pd
import pickle

from infoformulas_listcomp import IG
import numpy as np
from joblib import Parallel, delayed
from CSFSLoader import CSFSLoader
from analysis_std_drop import _conduct_analysis, visualise_results

N_features = [3,5,7,11,15,20]
N_features = [5,11,20]
N_samples = 100
dataset_name = 'Olympia_2_update'
dataset_name = 'olympia_subset1'
target = 'medals'
ignore_attributes = ['id']

def do_analysis():
    path = "datasets/olympia/{}.csv".format(dataset_name)
    df = CSFSLoader().load_dataset(path, ignore_attributes)

    Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, target, std, N_features, N_samples, dataset_name) for std in np.linspace(0.00001, .3, 100))

def evaluate():
    visualise_results(dataset_name, N_features, show_plot=False, N_samples=N_samples)


def explore():
    path = "datasets/olympia/Olympia_2_update.csv"
    df = CSFSLoader().load_dataset(path, ignore_attributes)
    target = "medals"
    print(df.head())
    print(df[target].describe())
    return
    df = df[:20]
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

def prepare_selected_dataset():
    csv_reader = csv.reader(open('datasets/olympia/olympia2_all_questions.csv','r', newline=''))
    features = [row[0] for row in csv_reader] #['electricity consumption_[16.0455, 20.243]_1', 'electricity consumption_[16.0455, 20.243]_0', 'electricity consumption_(24.87, 29.302]_1',...]
    for i in range(len(features)):
        features[i] = re.sub(r'_[01]$', '', features[i])
    print(features)
    path = "datasets/olympia/Olympia_2_update.csv"
    df = CSFSLoader().load_dataset(path)
    # print(df)
    df[features].to_csv('datasets/olympia/olympia_subset1.csv', sep=',', index=False)

def extract_prefix():
    file_name = "datasets/olympia/featuresOlympia_2_update.csv"
    reader = csv.reader(open(file_name,'r'), delimiter=",", quotechar='"')

    features = [(re.sub(r'_[\[\(\d].*?$','', row[0])) for row in reader]
    for f in set(features):
        print(f)

def explore_pickle():
    datasets = [
        'pickle-dumps/olympia_subset1/11features_100samples_0.000010000std.pickle',
        'pickle-dumps/olympia_subset1/11features_100samples_0.051523434std.pickle',
        'pickle-dumps/olympia_subset1/11features_100samples_0.093946263std.pickle'
    ]
    for ds in datasets:
        data = pickle.load(open(ds,'rb'))
        print(data['best_noisy_features_count'])
        print()

# do_analysis()
# evaluate()

# visualize_result()
# explore()
# extract_prefix()
# prepare_selected_dataset()
explore_pickle()
