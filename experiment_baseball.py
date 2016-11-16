import numpy as np
import pandas as pd
import re

import sys

from joblib import Parallel, delayed
from sklearn.preprocessing import Imputer, OneHotEncoder
from tabulate import tabulate

import CSFSLoader
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestActualSelector
from abstract_experiment import AbstractExperiment
from analysis_noisy_means_drop import _conduct_analysis, visualise_results
from imputer import CategoricalImputer
from infoformulas_listcomp import _H, IG_from_series, H
from util.util_features import get_feature_inf, get_features_from_questions


class ExperimentBaseball(AbstractExperiment):

    def __init__(self, dataset_name, experiment_number):
        super().__init__(dataset_name, experiment_number)
        self.path_raw = "datasets/baseball/raw/Baseball_original.csv"
        self.path_cleaned = "datasets/baseball/cleaned/experiment1/baseball_plus.csv"
        self.path_bin = "datasets/baseball/cleaned/experiment1/baseball_plus_bin.csv"
        self.path_meta = "datasets/baseball/cleaned/experiment1/baseball_plus_bin_meta.csv"
        self.path_questions = ""
        self.path_flock_result = ""
        self.target = "Rank"


    def preprocess_raw(self):
        """
        Selects only interesting features, fills gaps, binning and binarise
        outputs a csv into "cleaned" folder
        :return:
        """
        df_raw = pd.read_csv(self.path_raw)

        features_to_remove = ['DivWin',
                              'WCWin',
                              'HBP',
                              'SF',
                              'name',
                              'park',

                                # we have no feature descriptions for these features:
                              'franchID',
                              'teamIDBR',
                              'teamIDlahman45',
                              'teamIDretro'
                              ]

        # remove features that are for sure useless (like park) or have too many missing values (e.g. SF)
        df_raw = df_raw.drop(features_to_remove, axis='columns')

        # impute means, medians, default values according to notebook
        imputer_mean = Imputer(strategy='mean')
        imputer_most_frequent = CategoricalImputer()

        cols_imp_mean = ['Ghome', 'SO', 'SB', 'CS', 'DP', 'attendance'] # maybe improve DP and attendance by using neighbouring rows
        cols_imp_most_freq = ['lgID', 'LgWin', 'WSWin']

        def insert_values(df, columns, imputer):
            values = imputer.fit_transform(df[columns])
            df[columns] = values
            return df

        df_raw = insert_values(df_raw, cols_imp_mean, imputer_mean)
        # use integer values
        cols_int = ['Ghome', 'SO', 'SB', 'CS', 'DP', 'attendance']
        df_raw[cols_int] = df_raw[cols_int].astype('int')

        df_raw = insert_values(df_raw, cols_imp_most_freq, imputer_most_frequent)

        # divID: encode na as own category: U (Unknown)
        df_raw.loc[df_raw['divID'].isnull(), 'divID'] = 'U'

        # output
        df_raw.to_csv(self.path_cleaned, index=False)

    def bin_binarise(self):
        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """
        df = pd.read_csv(self.path_cleaned)

        def binarise_rank(row):
            df_prev = df[(df['yearID'] < row['yearID']) & (df['teamID'] == row['teamID'])]
            rank_rel = -1
            if len(df_prev) > 0:
                # print(df_prev[['yearID', 'teamID', 'Rank']])
                year_prev = max(df_prev['yearID'])
                rank_prev = df_prev[df_prev['yearID'] == year_prev]['Rank']
                rank_curr = row['Rank']
                rank_rel = rank_curr / rank_prev
            row['Rank_rel'] = float(rank_rel)
            return row
        df['Rank_rel'] = -1
        # df = df[:200]
        df = df.apply(binarise_rank, axis='columns')
        # filter out where we have no values
        df = df[df['Rank_rel'] != -1]
        # binarise target
        df[self.target] = df['Rank_rel'].apply(lambda x: 1 if x >= 1 else 0)
        # drop columns we dont need any more
        df = df.drop('Rank_rel', axis='columns')
        df = df.drop('teamID', axis='columns')
        # exit()
        # print('remove rows:',len(df[df['Rank_rel'] == -1]))
        # print('non changing:',len(df[df['Rank_rel'] == 1]))
        # print('higher:',len(df[df['Rank_rel']>1]))
        # print('lower:',len(df[df['Rank_rel']<1]))
        # print('all:',len(df))



        cols_binning = ['yearID', 'G', 'Ghome',
            'W', 'L', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB',
            'SO', 'SB', 'CS', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA',
            'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'attendance', 'BPF', 'PPF',
            ]

        for col in cols_binning:
            df[col] = pd.cut(df[col], 3)

        features = list(df.columns)
        features.remove(self.target) # do not bin target variable
        for col in features:
            dummies = pd.get_dummies(df[col], prefix='{}'.format(col))
            df = df.join(dummies)
            df = df.drop(col, axis='columns')
        # print(tabulate(df[:100], headers='keys'))

        df.to_csv(self.path_bin, index=False)

    def drop_analysis(self):
        N_features = [3, 5, 7, 10, 20, 30, 50, 70, 90]
        N_samples = 100
        df = CSFSLoader.CSFSLoader().load_dataset(self.path_bin)
        Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, self.target, mean_error, N_features, N_samples, self.dataset_name) for mean_error in np.linspace(0.0, 0.6, 200))

    def drop_evaluation(self):
         # example call
        N_features = [3, 5, 7, 10, 20, 30, 50, 70, 90]
        N_samples = 100
        visualise_results(dataset_name=self.dataset_name, N_features=N_features, show_plot=False, N_samples=N_samples, dataset_class='baseball', target=self.target)


    def evaluate_flock(self):
        df_data = self._get_dataset_bin() # use get_dataset() for original dataset
        # evaluator = CSFSEvaluator(df_data, self.target)
        #
        # R = range(3, len(df_data), 1) # number of samples
        # N_Feat = [3, 5, 7, 9, 11]
        # n_samples = 100 # number of repetitions to calculate average auc score for samples
        #
        # result = pd.DataFrame(columns=N_Feat, index=R)
        #
        # for r in R:
        #     sys.stdout.write('processing r =', r, '\n')
        #     aucs = {n_feat: list() for n_feat in N_Feat}
        #     for i in range(n_samples):
        #         # get a number of samples
        #         df_sample = df_data.sample(n=r, axis='rows')
        #         df_sample.index = range(r) # otherwise we have a problem with automatic iteration when calculating conditional probabilities
        #         best_selector = CSFSBestActualSelector(df_sample, self.target)
        #
        #         for n_feat in N_Feat:
        #             nbest_features = best_selector.select(n_feat)
        #             auc = evaluator.evaluate_features(nbest_features)
        #             aucs[n_feat].append(auc)
        #     result.loc[r] = {n_feat: np.mean(aucs[n_feat]) for n_feat in aucs}
        # result.to_csv(self.path_flock_result)

    # def evaluate_noisy_means_drop(self):
    #     N_features = []
    #     N_samples = 100
    #     df = CSFSLoader.CSFSLoader().load_dataset(self.path_bin, )
    #     Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, self.target, mean_error, N_features, N_samples, self.dataset_name) for mean_error in np.linspace(0.0, 0.6, 200))



if __name__ == '__main__':
    experiment = ExperimentBaseball('baseball', 1)
    # experiment.preprocess_raw()
    # experiment.bin_binarise()
    # experiment.get_metadata()
    experiment.drop_analysis()
    # experiment.evaluate_flock()