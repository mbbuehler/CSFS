import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from tabulate import tabulate

from abstract_experiment import AbstractExperiment
from infoformulas_listcomp import _H, IG_from_series
from util.util_features import get_feature_inf


class ExperimentOlympia(AbstractExperiment):

    def __init__(self, dataset_name, experiment_number):
        super().__init__(dataset_name, experiment_number)
        self.path_raw = 'datasets/olympia/raw/experiment3/Olympic2016_raw_allyears.csv'
        self.path_cleaned = 'datasets/olympia/cleaned/experiment3/olympic_allyears_plus.csv'
        self.path_bin = 'datasets/olympia/cleaned/experiment3/olympic_allyears_plus_bin.csv'
        self.path_meta = 'datasets/olympia/cleaned/experiment3/olympic_allyears_plus_bin_meta.csv'
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
                tmp_df = df[df[f]==cond_value]
                result.append(np.mean(tmp_df[target]))
            return result

        df['p|f=0'] = cond_mean(df_data, cond_value=0, target=self.target)
        df['p|f=1'] = cond_mean(df_data, cond_value=1, target=self.target)
        df['std'] = np.std(df_data)
        h_x = _H([df.loc[self.target]['p'], 1-df.loc[self.target]['p']])
        df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x)

        df.to_csv(self.path_meta, index=True)

if __name__ == '__main__':
    experiment = ExperimentOlympia('olympia', 3)
    # experiment.preprocess_raw()
    experiment.bin_binarise()
    experiment.get_metadata()