import pandas as pd


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
        pass

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