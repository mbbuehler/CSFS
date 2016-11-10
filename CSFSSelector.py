from random import shuffle
import numpy as np
from infoformulas_listcomp import IG, H, _H, H_cond, IG_fast
import pandas as pd


class CSFSSelector:

    def __init__(self, df, target, df_crowd = None):
        self.df = df
        self.target = target
        self.all_features = [f for f in df]
        self.all_predictors = [f for f in df if f != target]

    def select(self, n):
        raise Exception('Must implement subclass')

    def _check_predictors_length(self, n):
        if len(self.all_predictors) < n:
            raise Exception('n > than len(all_predictors). Not enough features available')

    @staticmethod
    def _get_ordered_predictors_dsc(dict_ig):
        return sorted(dict_ig, key=dict_ig.__getitem__, reverse=True)


class CSFSRandomSelector(CSFSSelector):

    def select(self, n):
        self._check_predictors_length(n)
        tmp_predictors = self.all_predictors.copy()
        shuffle(tmp_predictors)
        return tmp_predictors[:n]


class CSFSBestActualSelector(CSFSSelector):

    def __init__(self, df, target, df_crowd = None):
        super().__init__(df, target)
        self.dict_ig = self._get_dict_ig()
        self.ordered_predictors_dsc = self._get_ordered_predictors_dsc(self.dict_ig)

    def select(self, n):
        self._check_predictors_length(n)
        return self.ordered_predictors_dsc[:n]


    def _get_dict_ig(self):
        h_x = H(self.df[self.target])

        return {f: IG_fast(self.df[self.target], self.df[f], h_x) for f in self.all_predictors}


class CSFSBestUncertainSelector(CSFSSelector):

    def __init__(self, df, target, fix_std = None):
        super().__init__(df, target)
        self.df_info = self._get_df_info(df)
        self.df_actual = df
        self.fix_std = fix_std

    def _get_df_info(self, df):
        entropies = [H(df[f]) for f in self.all_features]
        mean = [np.mean(df[f]) for f in self.all_features]
        std = [np.std(df[f]) for f in self.all_features]

        cond_mean_target_f0 = [np.mean(df[df[f] == 0][self.target]) for f in self.all_features]

        cond_mean_target_f1= [np.mean(df[df[f] == 1][self.target]) for f in self.all_features]

        data = {'H': entropies, 'mean': mean, 'std': std, 'cond_mean_target_f0': cond_mean_target_f0, 'cond_mean_target_f1': cond_mean_target_f1}
        df_info = pd.DataFrame(data)
        df_info.index = self.all_features
        return df_info

    def _sample_noisy_x1_y1(self, pred):
        """

        :param pred:
        :return: a single noisy sample for the conditional mean p(x=1|y=1) Can be negative!
        """
        std = self.fix_std if self.fix_std is not None else self.df_info['std'].loc[pred]
        return np.random.normal(self.df_info['cond_mean_target_f1'].loc[pred], std)

    def _get_noisy_x1_y1(self):
        """
        Samples valid noisy conditional means p(x=1|y=1) (only allow values 0 <= p <= 1)
        :return: list with valid noisy conditional means for all features
        """
        x1_y1_list = []
        for pred in self.all_features:
            noisy_x1_y1 = self._sample_noisy_x1_y1(pred)
            while 0 > noisy_x1_y1 or 1 < noisy_x1_y1:
                noisy_x1_y1 = self._sample_noisy_x1_y1(pred)
            x1_y1_list.append(noisy_x1_y1)
        return x1_y1_list

    def _get_dict_ig(self):
        dict_ig = dict()
        self.df_info['cond_mean_target_f1noisy'] = self._get_noisy_x1_y1()

        h_x = _H([self.df_info['mean'].loc[self.target], 1 - self.df_info['mean'].loc[self.target]])
        for pred in self.all_predictors:
            h_cond = H_cond(self.df_info['cond_mean_target_f0'].loc[pred], self.df_info['cond_mean_target_f1noisy'].loc[pred], self.df_info['mean'].loc[pred])
            ig = h_x - h_cond
            dict_ig[pred] = ig
        return dict_ig

    def select(self, n):
        self._check_predictors_length(n)
        dict_ig = self._get_dict_ig()
        return self._get_ordered_predictors_dsc(dict_ig)[:n]

class CSFSAllFeaturesNoisySelector(CSFSSelector):

    def __init__(self, df, target, max_error): # max_error: max difference from mean e.g. 0.1
        super().__init__(df, target)
        self.df_info = self._get_df_info(df)
        self.df_actual = df
        self.max_error = max_error

    def _get_df_info(self, df):
        mean = [np.mean(df[f]) for f in self.all_features]
        cond_mean_target_f0 = [np.mean(df[df[f] == 0][self.target]) for f in self.all_features]
        cond_mean_target_f1= [np.mean(df[df[f] == 1][self.target]) for f in self.all_features]

        data = {'mean': mean, 'cond_mean_target_f0': cond_mean_target_f0, 'cond_mean_target_f1': cond_mean_target_f1}
        df_info = pd.DataFrame(data)
        df_info.index = self.all_features
        return df_info

    def _get_noisy_mean(self, mean, max_error):
        """
        Creates a noisy mean from mean within the range +- max_error
        :param mean: float 0<=mean<=1
        :param max_error: float 0<=max_error<=1 e.g. 0.1
        :return: float 0<=noisy_mean<=1
        """
        upper_lim = min(mean + max_error, 1)
        lower_lim = max(mean - max_error, 0)
        noisy_mean = (upper_lim - lower_lim) * np.random.random_sample() + lower_lim
        # print('mean', mean)
        # # print('upper', upper_lim)
        # # print('lower', lower_lim)
        # print('noisy mean', noisy_mean)
        return noisy_mean

    def _get_dict_ig(self):
        dict_ig = dict()
        noisy_mean_target = self._get_noisy_mean(self.df_info['mean'].loc[self.target], self.max_error)

        h_x = _H([noisy_mean_target, 1 - noisy_mean_target])
        for pred in self.all_predictors:
            h_cond = H_cond(
                self._get_noisy_mean(self.df_info['cond_mean_target_f0'].loc[pred], self.max_error),
                self._get_noisy_mean(self.df_info['cond_mean_target_f1'].loc[pred], self.max_error),
                self._get_noisy_mean(self.df_info['mean'].loc[pred], self.max_error)
            )
            ig = h_x - h_cond
            dict_ig[pred] = ig
        return dict_ig

    def select(self, n):
        self._check_predictors_length(n)
        dict_ig = self._get_dict_ig()
        return self._get_ordered_predictors_dsc(dict_ig)[:n]

def test():
    n = 4
    data = {'F1': [1,1,1,1,0,0,0], 'F2': [1,1,0,0,1,1,1], 'F3': [0,0,0,0,1,1,1], 'F4': [1,1,1,1,1,1,1], 'T': [1,0,1,0,1,0,1]}
    df = pd.DataFrame(data)
    target = 'T'
    bestUncertainSelector = CSFSBestUncertainSelector(df, target, fix_std=0.1)
    bestNoisyFeatures = bestUncertainSelector.select(n)
    print(bestNoisyFeatures)

    allFeaturesNoisySelector = CSFSAllFeaturesNoisySelector(df, target, max_error=0.1)
    allFeaturesNoisy = allFeaturesNoisySelector.select(n)
    print(allFeaturesNoisy)

if __name__ == '__main__':
    test()