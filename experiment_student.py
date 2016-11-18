import numpy as np
import pandas as pd
import re

import sys

from sklearn.preprocessing import Imputer
from tabulate import tabulate

from CSFSCrowdCleaner import CSFSCrowdAggregator, CSFSCrowdAnalyser, CSFSCrowdCleaner
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestActualSelector, CSFSBestFromMetaSelector
from abstract_experiment import AbstractExperiment
from infoformulas_listcomp import _H, IG_from_series, H
from util.util_features import get_feature_inf, get_features_from_questions


class ExperimentStudent(AbstractExperiment):

    def __init__(self, dataset_name, experiment_number):
        super().__init__(dataset_name, experiment_number)
        experiment_name = 'experiment1'

        self.path_raw = '{}raw/{}/student-comb.csv'.format(self.base_path, experiment_name)
        self.path_cleaned = '{}cleaned/{}/student-comb_clean.csv'.format(self.base_path, experiment_name)
        self.path_bin = '{}cleaned/{}/student-comb_clean_bin.csv'.format(self.base_path, experiment_name)
        self.path_meta = '{}cleaned/{}/student-comb_clean_bin_meta.csv'.format(self.base_path, experiment_name)
        # self.path_answers_raw = '{}results/{}/answers_raw.xlsx'.format(base_path, experiment_name)
        # self.path_answers_clean = '{}results/{}/answers_clean.csv'.format(base_path, experiment_name)
        # self.path_answers_aggregated = '{}results/{}/answers_aggregated.csv'.format(base_path, experiment_name)
        # self.path_answers_metadata = '{}results/{}/answers_metadata.csv'.format(base_path, experiment_name)
        # self.path_csfs_auc = '{}results/{}/csfs_auc.csv'.format(base_path, experiment_name)
        # self.path_questions = '{}questions/{}/questions.csv'.format(base_path, experiment_name) # experiment2 for experiment3
        # self.path_flock_result = '{}questions/{}/flock_auc.csv'.format(experiment_name)
        self.target = 'G3'


    def preprocess_raw(self):
        """
        Selects only interesting features, fills gaps
        outputs a csv into "cleaned" folder
        :return:
        """
        df_raw = pd.read_csv(self.path_raw)
        features_to_remove = ['G1', 'G2']
        # only take subset we have questions for
        df_raw = df_raw.drop(features_to_remove, axis='columns')
        df_raw.to_csv(self.path_cleaned, index=False)

    def bin_binarise(self):
        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """

        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """
        df = pd.read_csv(self.path_cleaned)
        target_median = np.median(df[self.target])

        df[self.target] = df[self.target].apply(lambda x: 1 if x > target_median else 0) # 1:"belongs to the better one" 0: "belongs to the lower half or middle"

        df['sex'] = df['sex'].apply(lambda x: 1 if x == 'M' else 0)
        df['school'] = df['school'].apply(lambda x: 1 if x == 'GP' else 0)
        df['address'] = df['address'].apply(lambda x: 1 if x == 'U' else 0)
        df['famsize'] = df['famsize'].apply(lambda x: 1 if x == 'GT3' else 0)
        df['Pstatus'] = df['Pstatus'].apply(lambda x: 1 if x == 'T' else 0)

        features_yesno = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
        def binarise_yesno(df, feature):
            df[feature] = df[feature].apply(lambda x: 1 if x == 'yes' else 0)
            return df
        for feature in features_yesno:
            df = binarise_yesno(df, feature)


        cols_nominal = ['Mjob', 'Fjob', 'reason', 'guardian']
        for col in cols_nominal:
            dummies = pd.get_dummies(df[col], prefix='{}'.format(col))
            df = df.join(dummies)
            df = df.drop(col, axis='columns')

        print(tabulate(df, headers='keys'))

        df.to_csv(self.path_bin, index=False)
        return

        features_to_bin = ['age',
                           '',
                           '']


        exit()
        # filter out where we have no values
        df = df[df['Rank_rel'] != -1]
        # binarise target
        df[self.target] = df['Rank_rel'].apply(lambda x: 1 if x > 1 else 0)
        # drop columns we dont need any more
        df = df.drop('Rank_rel', axis='columns')
        df = df.drop('teamID', axis='columns')

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

    def get_metadata(self):
        """
        Outputs a csv with p, p|f=0, p|f=1, H, Ig, Ig ratio in "cleaned" folder
        :return:
        """
        df_data = pd.read_csv(self.path_bin)
        df_data = df_data[df_data['subject']==1]

        df = pd.DataFrame()
        df['p'] = np.mean(df_data)

        def cond_mean(df, cond_value, target):
            result = list()
            for f in df:
                tmp_df = df[df[f] == cond_value]
                result.append(np.mean(tmp_df[target]))
            return result

        df['p|f=0'] = cond_mean(df_data, cond_value=0, target=self.target)
        df['p|f=1'] = cond_mean(df_data, cond_value=1, target=self.target)
        df['std'] = np.std(df_data)

        df['H'] = [H(df_data[x]) for x in df_data]
        h_x = _H([df.loc[self.target]['p'], 1-df.loc[self.target]['p']])
        df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x)
        df['IG ratio'] = df.apply(lambda x: x['IG']/x['H'], axis='columns') # correct?
        df.to_csv(self.path_meta, index=True)

    def _get_dataset_bin(self):
        """
        Selects subset of data set we have questions for.
        """
        df_raw = pd.read_csv(self.path_bin)
        # kick all features we don't want
        features = get_features_from_questions(self.path_questions, remove_cond=True)
        features.append(self.target)
        df = df_raw[features]
        return df

    def evaluate_crowd_all_answers(self):
        """
        Aggregates crowd answers and evaluates for all crowd answers
        :return:
        """
        df_clean = CSFSCrowdCleaner(self.path_questions, self.path_answers_raw, self.target).clean()
        df_clean.to_csv(self.path_answers_clean, index=True)

        df_aggregated = CSFSCrowdAggregator(df_clean).aggregate()
        df_aggregated.to_csv(self.path_answers_aggregated, index=True)

        df_combined = CSFSCrowdAnalyser().get_combined_df(self.path_answers_aggregated, self.path_meta)
        df_combined.to_csv(self.path_answers_metadata, index=True)


    def evaluate_csfs_auc(self):
        df_data = self._get_dataset_bin()
        evaluator = CSFSEvaluator(df_data, self.target)
        R = range(3, len(df_data), 1) # number of samples
        N_Feat = [3, 5, 7, 9, 11]
        result = pd.DataFrame(columns=N_Feat, index=R)

        df_aggregated = pd.read_csv(self.path_answers_aggregated, index_col=0)
        selector = CSFSBestFromMetaSelector(df_aggregated)

        aucs = dict()

        for n_feat in N_Feat:
            nbest_features = selector.select(n_feat)
            auc = evaluator.evaluate_features(nbest_features)
            aucs[n_feat] = auc

        # we are not sure how many answers we have for each feature. -> fill a dataframe with constant values.
        answer_count_min = df_aggregated['n p'].min().astype('int')
        answer_count_max = df_aggregated['n p'].max().astype('int')

        df_csfs_auc = pd.DataFrame(aucs, index=range(answer_count_min, answer_count_max))

        df_csfs_auc.to_csv(self.path_csfs_auc)


    def evaluate_flock(self):
        df_data = self._get_dataset_bin()
        evaluator = CSFSEvaluator(df_data, self.target)

        R = range(3, len(df_data), 1) # number of samples
        N_Feat = [3, 5, 7, 9, 11]
        n_samples = 100 # number of repetitions to calculate average auc score for samples

        result = pd.DataFrame(columns=N_Feat, index=R)

        for r in R:
            sys.stdout.write('processing r =', r, '\n')
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

    # def evaluate_noisy_means_drop(self):
    #     N_features = []
    #     N_samples = 100
    #     df = CSFSLoader.CSFSLoader().load_dataset(self.path_bin, )
    #     Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, self.target, mean_error, N_features, N_samples, self.dataset_name) for mean_error in np.linspace(0.0, 0.6, 200))



if __name__ == '__main__':
    experiment = ExperimentStudent('student', 1)
    # experiment.set_up_basic_folder_structure()
    # experiment.set_up_experiment_folder_structure('experiment1')
    experiment.preprocess_raw()
    experiment.bin_binarise()
    experiment.get_metadata()
    # experiment.evaluate_crowd_all_answers()
    # experiment.evaluate_flock()
    # experiment.evaluate_csfs_auc()