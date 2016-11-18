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
from util.util_features import get_feature_inf, get_features_from_questions


class ExperimentOlympia(AbstractExperiment):

    def __init__(self, dataset_name, experiment_number):
        super().__init__(dataset_name, experiment_number)
        experiment_name = 'experiment3'
        experiment_name = 'experiment4_extreme-cond-means'
        experiment_name = 'experiment2-4_all'

        self.path_raw = 'datasets/olympia/raw/{}/Olympic2016_raw_allyears.csv'.format(experiment_name)
        self.path_cleaned = 'datasets/olympia/cleaned/{}/olympic_allyears_plus.csv'.format(experiment_name)
        self.path_bin = 'datasets/olympia/cleaned/{}/olympic_allyears_plus_bin.csv'.format(experiment_name)
        self.path_meta = 'datasets/olympia/cleaned/{}/olympic_allyears_plus_bin_meta.csv'.format(experiment_name)
        self.path_answers_raw = 'datasets/olympia/results/{}/answers_raw.xlsx'.format(experiment_name)
        self.path_answers_clean = 'datasets/olympia/results/{}/answers_clean.csv'.format(experiment_name)
        self.path_answers_aggregated = 'datasets/olympia/results/{}/answers_aggregated.csv'.format(experiment_name)
        self.path_answers_metadata = 'datasets/olympia/results/{}/answers_metadata.csv'.format(experiment_name)
        self.path_csfs_auc = 'datasets/olympia/results/{}/csfs_auc.csv'.format(experiment_name)
        self.path_questions = 'datasets/olympia/questions/{}/questions.csv'.format(experiment_name) # experiment2 for experiment3
        self.path_flock_result = 'datasets/olympia/questions/{}/flock_auc.csv'.format(experiment_name)
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

    def get_metadata(self):
        """
        Outputs a csv with p, p|f=0, p|f=1, H, Ig, Ig ratio in "cleaned" folder
        :return:
        """
        df_data = pd.read_csv(self.path_bin)

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
    experiment = ExperimentOlympia('olympia', 4)
    # experiment.preprocess_raw()
    # experiment.bin_binarise()
    # experiment.get_metadata()
    experiment.evaluate_crowd_all_answers()
    # experiment.evaluate_flock()
    # experiment.evaluate_csfs_auc()