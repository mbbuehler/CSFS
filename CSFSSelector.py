from random import shuffle
import numpy as np
from infoformulas_listcomp import IG, H, _H
import pandas as pd

from noise_helper_funcs import H_cond


class CSFSSelector:

    def __init__(self, df, target, df_crowd = None):
        self.target = target
        self.all_features = [f for f in df]
        self.all_predictors = [f for f in df if f != target]
        if df_crowd:
            self.df_crowd = df_crowd

    def select(self, n):
        raise Exception('Must implement subclass')

    def _check_predictors_length(self, n):
        if len(self.all_predictors) < n:
            raise Exception('n > than len(all_predictors). Not enough features available')


class CSFSRandomSelector(CSFSSelector):

    def select(self, n):
        self._check_predictors_length(n)
        tmp_predictors = self.all_predictors.copy()
        shuffle(tmp_predictors)
        return tmp_predictors[:n]


class CSFSBestActualSelector(CSFSSelector):

    def __init__(self, df, target, df_crowd = None):
        super().__init__(df, target)
        self.dict_ig = self._get_dict_ig(df, self.target)
        self.ordered_predictors_dsc = self._get_orderend_predictors_dsc(self.dict_ig)

    def select(self, n):
        self._check_predictors_length(n)
        return self.ordered_predictors_dsc[:n]

    def _get_dict_ig(self, df, target):
        return {f: IG(df[target], df[f]) for f in self.all_predictors}

    def _get_orderend_predictors_dsc(self, dict_ig):
        return sorted(dict_ig, key=dict_ig.__getitem__, reverse=True)

class CSFSBestUncertainSelector(CSFSSelector):

    def __init__(self, df, target, df_crowd = None):
        super().__init__(df, target)
        self.df_info = self._get_df_info(df)
        self.df_actual = df
        print(self.df_info)
        # if df_crowd is None:
        #     raise Exception('Need a df crowd!')

    def _get_df_info(self, df):
        entropies = [H(df[f]) for f in self.all_features]
        mean = [np.mean(df[f]) for f in self.all_features]
        std = [np.std(df[f]) for f in self.all_features]
        tmp_df = df[df[self.target] == 0]
        cond_mean_f_0 = [np.mean(tmp_df[f]) for f in self.all_features]
        tmp_df = df[df[self.target] == 1]
        cond_mean_f_1 = [np.mean(tmp_df[f]) for f in self.all_features]
        data = {'H': entropies, 'mean': mean, 'std': std, 'cond_mean_f_0': cond_mean_f_0, 'cond_mean_f_1': cond_mean_f_1}
        df_info = pd.DataFrame(data)
        df_info.index = self.all_features
        return df_info

    def _sample_noisy_x1_y1(self, pred):
        return np.random.normal(self.df_info['cond_mean_f_1'].loc[pred], self.df_info['std'].loc[pred], 1)

    def _get_noisy_x1_y1(self):
        x1_y1_list = []
        for pred in self.all_predictors:
            noisy_x1_y1 = self._sample_noisy_x1_y1(pred)
            while 0 > noisy_x1_y1 or 1 < noisy_x1_y1:
                noisy_x1_y1 = self._sample_noisy_x1_y1(pred)
            x1_y1_list.append(noisy_x1_y1)
        return x1_y1_list

    def _get_dict_ig(self):
        self.df_info['cond_mean_f1_noisy'] = self._get_noisy_x1_y1()
        self.df_info['h_cond'] = [H_cond(self.df_info['cond_mean_f_0'].loc[pred], self.df_info['cond_mean_f1_noisy'].loc[pred], self.df_info['y1'].loc[pred]) for pred in self.all_predictors]
        self.df_info['h_x'] = [_H([self.df_info['mean'].loc[pred], 1 - self.df_info['mean'].loc[pred]]) for pred in self.all_predictors]
        # print(self.df_info)

    def select(self, n):
        self._get_dict_ig()


        # print(x1_y1)
        # if 0 <= x1_y1 <= 1: # make sure we have a valid probability
        #     h_cond = H_cond(df_actual.loc[feature]['cond_mean_f_0'], x1_y1, df_actual.loc[feature]['mean'])
        #     h_x = _H([x1, 1-x1])
        #     igs.append(inf_gain(h_x, h_cond))
        #
        pass


# for x1_y1 in np.random.normal(df_actual.loc[feature]['cond_mean_f_1'], df_actual.loc[feature]['std'], N_noise):
#         if 0 <= x1_y1 <= 1: # make sure we have a valid probability
#             h_cond = H_cond(df_actual.loc[feature]['cond_mean_f_0'], x1_y1, df_actual.loc[feature]['mean'])
#             h_x = _H([x1, 1-x1])
#             igs.append(inf_gain(h_x, h_cond))

