import numpy as np
import pandas as pd

from infoformulas_listcomp import H, _H, IG_from_series


class AbstractExperiment():

    def __init__(self, dataset_name, experiment_number):
        self.dataset_name = dataset_name
        self.number = experiment_number

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


    """
    - get crowd answers + aggregate
    - flock experiment
    - std noise experiment
    """