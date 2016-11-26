import os
import numpy as np
import pandas as pd
import sys

from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestActualSelector
from infoformulas_listcomp import H, _H, IG_from_series
from util.util_features import get_features_from_questions


class AbstractExperiment:
    def __init__(self, dataset_name, experiment_number, experiment_name):
        self.dataset_name = dataset_name
        self.number = experiment_number
        self.experiment_name = experiment_name
        self.base_path = 'datasets/{}/'.format(self.dataset_name)

    def _create_if_nonexisting(self, path, folder):
            if folder not in os.listdir(path):
                os.mkdir('{}{}'.format(path, folder))

    def set_up_basic_folder_structure(self):
        default_folders = ['cleaned', 'raw', 'questions', 'results']

        for folder in default_folders:
            self._create_if_nonexisting(self.base_path, folder)

    def set_up_experiment_folder_structure(self, experiment_name):
        default_folders = ['cleaned', 'raw', 'questions', 'results']

        folder_name = experiment_name

        for folder in default_folders:
            path = '{}{}/'.format(self.base_path, folder)
            self._create_if_nonexisting(path, folder_name)

        self._create_if_nonexisting('{}raw/'.format(self.base_path), 'default')



    def explore_original(self):
        """
        Outputs a python notebook with H, Ig, Ig ratio in "raw" folder
        :return:
        """
        pass

    def preprocess_raw(self):
        """
        Selects only interesting features, fills gaps
        outputs a csv into "cleaned" folder "_clean"
        :return:
        """
        pass

    def bin_binarise(self):
        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """
        pass

    def get_metadata(self):
        """
        Outputs a csv with p, p|f=0, p|f=1, H, Ig, Ig ratio in "cleaned" folder
        :return:
        """
        df_data = pd.read_csv(self.path_bin)
        # df_data = df_data[df_data['subject'] == 0] # limit data to certain subject.

        df = pd.DataFrame()
        df['mean'] = np.mean(df_data)

        def cond_mean(df, cond_value, target):
            result = list()
            for f in df:
                tmp_df = df[df[f] == cond_value]
                result.append(np.mean(tmp_df[target]))
            return result

        df['mean|f=0'] = cond_mean(df_data, cond_value=0, target=self.target)
        df['mean|f=1'] = cond_mean(df_data, cond_value=1, target=self.target)
        df['std'] = np.std(df_data)

        df['H'] = [H(df_data[x]) for x in df_data]
        h_x = _H([df.loc[self.target]['mean'], 1-df.loc[self.target]['mean']])
        df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='mean')
        df['IG ratio'] = df.apply(lambda x: x['IG']/x['H'], axis='columns') # correct?
        df = df.sort_values(by=['IG'], ascending=False)
        df.to_csv(self.path_meta, index=True)

    def _remove_non_informative_rows(self, df, threshold):
        """
        returns row indices where more than threshold entries are missing, e.g. 0.5
        :param threshold: ratio, e.g. 0.5
        """
        df_tmp = pd.DataFrame()
        n_features = len(df.columns)
        # calculating ratio of rows that have more than "ratio" missing values
        df_tmp['ratio'] = df.apply(lambda row: row.isnull().sum()/n_features, axis='columns')

        # kick too noisy rows
        return df[df_tmp['ratio'] <= threshold]

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

    def evaluate_flock(self):
        df_data = self._get_dataset_bin()
        evaluator = CSFSEvaluator(df_data, self.target)

        R = range(3, 100, 1) # number of samples
        N_Feat = [3, 5, 7, 9, 11]
        n_samples = 1 # number of repetitions to calculate average auc score for samples

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

