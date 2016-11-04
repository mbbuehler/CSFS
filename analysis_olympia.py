import csv
import re

import pandas as pd
import pickle

from tabulate import tabulate

from infoformulas_listcomp import IG, _H, IG_from_series
import numpy as np
from joblib import Parallel, delayed
from CSFSLoader import CSFSLoader
from analysis_noisy_means_drop import _conduct_analysis, visualise_results

N_features = [3,5,7,11,15,20]
N_samples = 100
dataset_name = 'Olympia_2_update'
dataset_name = 'olympia_subset1'
target = 'medals'
ignore_attributes = ['id']

def do_analysis():
    path = "datasets/olympia/{}.csv".format(dataset_name)
    df = CSFSLoader().load_dataset(path)

    Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, target, std, N_features, N_samples, dataset_name) for std in np.linspace(0.00001, .6, 200))

def evaluate():
    N_features = [3,5,7,11]
    visualise_results(dataset_name=dataset_name, N_features=N_features, show_plot=False, N_samples=N_samples, dataset_class='olympia', target=target)


def explore():
    path = "datasets/olympia/cleaned/experiment2/Olympic2016_raw_plus.csv"
    df = pd.read_csv(path)
    df = df.dropna()
    for f in sorted([f for f in df]):
        print(f, min(df[f]), max(df[f]))

def explore_feature(feature):
    path = "datasets/olympia/Olympia_2_update.csv"
    df = CSFSLoader().load_dataset(path, ignore_attributes)
    print(df[feature].describe())


def prepare_selected_dataset():
    csv_reader = csv.reader(open('datasets/olympia/olympia2_all_questions.csv','r', newline=''))
    features = [row[0] for row in csv_reader] #['electricity consumption_[16.0455, 20.243]_1', 'electricity consumption_[16.0455, 20.243]_0', 'electricity consumption_(24.87, 29.302]_1',...]
    for i in range(len(features)):
        features[i] = re.sub(r'_[01]$', '', features[i])
    # print(features)
    # print(len(features))
    # print(list(set(features)))
    # print(len(list(set(features))))
    # return
    features = list(set(features))
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
        'pickle-dumps/olympia_subset1/11features_100samples_0.000error.pickle',
        'pickle-dumps/olympia_subset1/11features_100samples_0.052error.pickle',
        'pickle-dumps/olympia_subset1/11features_100samples_0.100error.pickle'
    ]
    for ds in datasets:
        data = pickle.load(open(ds,'rb'))
        print(data['best_noisy_mean_features_count'])
        print()

def explore_olympia_bin_features():
    features = [
        'education expenditures_(4.133, 5.6]',
        'region_3',
        'internet users_[14400, 1167666.667]',
        'electricity consumption per capita_(4244.0286, 48501.33]',
        'medals'
        ]
    base_path = '/home/marcello/studies/bachelorarbeit/workspace/github_crowd-sourcing-for-feature-selection/datasets/olympia/cleaned/experiment2/'
    path = base_path+'Olympic2016_raw_plus_bin.csv'
    df_data = CSFSLoader().load_dataset(path)
    # df_data = df_data[features]
    df = pd.DataFrame()
    df['p'] = np.mean(df_data)

    # print(df_data)

    def cond_mean(df, cond_value):
        result = list()
        for f in df:
            tmp_df = df[df[f]==cond_value]
            result.append(np.mean(tmp_df['medals']))
        return result

    df['p|f=0'] = cond_mean(df_data, cond_value=0)
    df['p|f=1'] = cond_mean(df_data, cond_value=1)
    df['std'] = np.std(df_data)
    h_x = _H([df.loc[target]['p'], 1-df.loc[target]['p']])
    df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x)


    path = base_path+'Olympic2016_raw_plus_bin_metadata.csv'
    df.to_csv(path, index=True)


# do_analysis()
# evaluate()
# explore_pickle()

# visualize_result()
explore()
# extract_prefix()
# prepare_selected_dataset()

# feature = 'electricity consumption_[16.0455, 20.243]'
# explore_feature(feature)
# explore_olympia_bin_features()
