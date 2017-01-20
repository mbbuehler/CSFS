from random import shuffle

import numpy as np
import pandas as pd
import re

import sys

from sklearn.preprocessing import Imputer
from CSFSCrowdCleaner import CSFSCrowdAggregator, CSFSCrowdAnalyser, CSFSCrowdCleaner
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestActualSelector, CSFSBestFromMetaSelector
from abstract_experiment import AbstractExperiment
from infoformulas_listcomp import _H, IG_from_series, H


class ExperimentOlympia(AbstractExperiment):

    def __init__(self, dataset_name, experiment_number, experiment_name):
        super().__init__(dataset_name, experiment_number, experiment_name)
        experiment_name = 'experiment3'
        experiment_name = 'experiment4_extreme-cond-means'
        experiment_name = 'experiment2-4_all'

        self.path_raw = 'datasets/olympia/raw/{}/Olympic2016_raw_allyears.csv'.format(experiment_name)
        self.path_cleaned = 'datasets/olympia/cleaned/{}/olympic_allyears_plus.csv'.format(experiment_name)
        self.path_bin = 'datasets/olympia/cleaned/{}/olympic_allyears_plus_bin.csv'.format(experiment_name)
        self.path_autocorrelation = '{}cleaned/{}/olympic_bin_autocorrelation.csv'.format(self.base_path, experiment_name)
        self.path_meta = 'datasets/olympia/cleaned/{}/olympic_allyears_plus_bin_meta.csv'.format(experiment_name)
        self.path_answers_raw = 'datasets/olympia/results/{}/answers_raw_mod.xlsx'.format(experiment_name)
        self.path_answers_clean = 'datasets/olympia/results/{}/answers_clean.csv'.format(experiment_name)
        self.path_answers_clean_grouped = '{}results/{}/answers_clean_grouped.pickle'.format(self.base_path, experiment_name)
        self.path_answers_plots = '{}results/{}/visualisations/'.format(self.base_path, experiment_name)
        self.path_answers_aggregated = 'datasets/olympia/results/{}/answers_aggregated.csv'.format(experiment_name)
        self.path_answers_metadata = 'datasets/olympia/results/{}/answers_metadata.csv'.format(experiment_name)
        self.path_csfs_auc = 'datasets/olympia/results/{}/csfs_auc.csv'.format(experiment_name)
        self.path_csfs_std = 'datasets/olympia/results/{}/csfs_std.csv'.format(experiment_name)
        self.path_questions = 'datasets/olympia/questions/{}/questions_mod2.csv'.format(experiment_name) # experiment2 for experiment3
        self.path_flock_result = 'datasets/olympia/results/{}/flock_auc.csv'.format(experiment_name)
        self.path_cost_ig_test = 'application/conditions/test/olympia.csv'
        self.path_cost_ig_expert = 'application/conditions/expert/olympia.csv'
        # self.path_budget_evaluation = 'datasets/olympia/budget/{}/budget_evaluation.csv'.format(experiment_name)


        self.path_budget_evaluation_cost = '{}evaluation/budget_evaluation_cost.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_nofeatures = '{}evaluation/budget_evaluation_nofeatures.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_cost_rawaucs = '{}evaluation/budget_evaluation_cost_rawaucs.pickle'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_nofeatures_rawaucs = '{}evaluation/budget_evaluation_nofeatures_rawaucs.pickle'.format(self.base_path, experiment_name)
        # self.path_budget_evaluation_result_domain = '{}evaluation/experts_domain/result_domain.csv'.format(self.base_path)
        # self.path_cost_ig_base = '{}evaluation/base.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_base = '{}evaluation/base.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_result = '{}evaluation/result.csv'.format(self.base_path, experiment_name)

        self.path_final_evaluation_aucs = '{}evaluation/final_evaluation_aucs.pickle'.format(self.base_path)
        self.path_final_evaluation_aggregated = '{}evaluation/final_evaluation_aggregated.pickle'.format(self.base_path)

        self.target = 'medals'


    def preprocess_raw(self):
        """
        Selects only interesting features, fills gaps, binning and binarise
        outputs a csv into "cleaned" folder
        :return:
        """
        df_raw = pd.read_csv(self.path_raw)

        features_selected = ['education expenditures',
                             'electricity consumption per person',
                             'electricity consumption',
                             'exports',
                             'inflation rate',
                             'internet users',
                             'public debt',
                             'region',
                             'unemployment rate',
                             'medals'
                             ]
        # only take subset we have questions for
        df_raw = df_raw[features_selected]
        # kick features with too many missing values
        df_raw = self._remove_non_informative_rows(df_raw, 0.5)

        # fill missing values
        imputer = Imputer()
        vals = imputer.fit_transform(df_raw)
        df_raw = pd.DataFrame(vals, columns=df_raw.columns, index=df_raw.index)

        # output
        df_raw.to_csv(self.path_cleaned, index=False)

    def bin_binarise(self):
        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """
        df = pd.read_csv(self.path_cleaned)
        df['medals'] = df['medals'] > 0
        df['medals'] = df['medals'].astype(int)

        binary_features = ['medals']
        features_to_bin = ['region']

        features_to_do = [f for f in df if f not in binary_features and f not in features_to_bin]
        # binarise
        for f in features_to_bin:
            df = df.combine_first(pd.get_dummies(df[f], prefix=f))
            df = df.drop(f, axis='columns')
        df.columns = [re.sub(r'\.[01]','',f) for f in df]

        feature_inf = get_feature_inf()
        def add_bin_data(df, f, feature_inf):
            """
            converts numerical to binary for given categories
            :param df:
            :param f:
            :param feature_inf:
            :return:
            """
            for bin in feature_inf[f]:
                print(bin)
                df[bin] = 0

            for bin in feature_inf[f]:
                lim_lower = feature_inf[f][bin][0]
                lim_upper = feature_inf[f][bin][1]

                # print(lim_lower, '< x <=', lim_upper, '?')
                selected_index = df[(df[f] > lim_lower) & (df[f] < lim_upper)].index#  & (df_tmp[f] <= lim_upper)
                df.loc[selected_index, bin] = 1
            df = df.drop(f, axis='columns')
            return df

        for f in features_to_do:
            df = add_bin_data(df, f, feature_inf)

        df.to_csv(self.path_bin, index=False)

    def evaluate_flock(self):
        df_data = self._get_dataset_bin()
        evaluator = CSFSEvaluator(df_data, self.target)

        R = range(3, len(df_data), 1) # number of samples
        N_Feat = [3, 5, 7, 9, 11]
        n_samples = 100 # number of repetitions to calculate average auc score for samples

        result = pd.DataFrame(columns=N_Feat, index=R)

        for r in R:
            sys.stdout.write('r: {}\n'.format(r))
            aucs = {n_feat: list() for n_feat in N_Feat}
            for i in range(n_samples):
                # get a number of samples
                df_sample = df_data.sample(n=r, axis='rows')
                df_sample.index = range(r) # otherwise we have a problem with automatic iteration when calculating conditional probabilities
                best_selector = CSFSBestActualSelector(df_sample, self.target)

                for n_feat in N_Feat:
                    nbest_features = best_selector.select(n_feat)
                    auc = evaluator.evaluate_features(nbest_features)
                    aucs[n_feat].append(auc)
            result.loc[r] = {n_feat: np.mean(aucs[n_feat]) for n_feat in aucs}
        result.to_csv(self.path_flock_result)


if __name__ == '__main__':
    experiment = ExperimentOlympia('olympia', 4, 'experiment2-4_all')
    # experiment.preprocess_raw()
    # experiment.bin_binarise()
    # experiment.get_metadata()
    # experiment.evaluate_crowd_all_answers()
    # experiment.evaluate_flock()
    # experiment.evaluate_csfs_auc(fake_till_n=25)


    budget_range = range(10, 160, 10)
    no_features = range(1, 14)
    # experiment.evaluate_ranking_cost(budget_range)
    # experiment.evaluate_ranking_nofeatures(no_features)
    # experiment.evaluate_budget(budget_range)
    # experiment.autocorrelation()
    experiment.final_evaluation(no_features, bootstrap_n=12, repetitions=20)
    experiment.final_evaluation_visualisation(no_features)
    # experiment.crowd_answers_plot()