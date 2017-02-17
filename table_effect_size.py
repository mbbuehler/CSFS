import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

from application.EvaluationRanking import ERCondition
from csfs_stats import hedges_g


class EffectSizeMatrix:
    def __init__(self, df, conditions, feature_slice=6, rename_columns=False, remove_null=True):
        self.df = df
        self.conditions = conditions
        self.feature_slice = feature_slice
        self.rename_columns = rename_columns
        self.remove_null = remove_null

    @staticmethod
    def _get_asteriks(p):
            asteriks = "+"
            if p <= 0.05:
                asteriks = "*"
            if p <= 0.01:
                asteriks = "**"
            if p <= 0.001:
                asteriks = "***"
            return asteriks

    def get_result_df(self):
        """
        Creates a df table comparing human conditions with the welch's t-test
        :param feature_slice: int
        :return:
        """
        aucs_filtered = {condition: self.df[condition][self.feature_slice] for condition in self.conditions}
        # print(aucs_filtered) # condition: [AUC]
        df_matrix = pd.DataFrame(columns=self.conditions, index=self.conditions)
        for cond1 in self.conditions:
            for cond2 in self.conditions:
                a = aucs_filtered[cond1]
                b = aucs_filtered[cond2]
                t, p = scipy.stats.ttest_ind(a, b, equal_var=False)
                g = hedges_g(a, b) # effect size
                value = "{:.3f}\textsuperscript{{{}}}".format(g, self._get_asteriks(p))
                df_matrix.loc[cond1, cond2] = value

        df_result = pd.DataFrame(np.triu(df_matrix.values, k=1), columns=self.conditions, index=self.conditions)
        if self.rename_columns:
            df_result = self.add_labels(df_result)
        if self.remove_null:
            df_result = self.remove_null_rows(df_result)
        return df_result

    def add_labels(self, df):
        df.columns = [ERCondition.get_string_paper(c) for c in df.columns]
        df.index = [ERCondition.get_string_paper(c) for c in df.index]
        return df

    def remove_null_rows(self, df):
        df = df[df.columns[1:]]
        df = df.iloc[:-1]
        df = df.apply(lambda r: r.apply(lambda v: "" if v==0 else v)) # better: compare index and column for equality
        return df

    def get_latex(self):
        """
        Creates a latex table comparing human conditions with the welch's t-test
        :param feature_slice: int
        :return:
        """
        df_result = self.get_result_df()
        return df_result.to_latex(escape=False)

class EffectSizeTable:
    def __init__(self, df, feature_range, conditions=['Human', 'KrowDD']):
        self.df = df
        self.conditions = conditions
        self.feature_range = feature_range

    @staticmethod
    def _get_asteriks(p):
            asteriks = "+"
            if p <= 0.05:
                asteriks = "*"
            if p <= 0.01:
                asteriks = "**"
            if p <= 0.001:
                asteriks = "***"
            return asteriks

    def get_result_series(self, dataset_name, condition_other, condition_better='KrowDD'):
        """
        Creates a df table comparing human conditions with the welch's t-test
        :param feature_slice: int
        :return:
        """
        aucs_filtered = {condition: self.df.loc[self.feature_range, condition] for condition in self.conditions}
        # print(aucs_filtered) # condition: [AUC]
        def get_value(a, b):
            t, p = scipy.stats.ttest_ind(a, b, equal_var=False)
            g = hedges_g(a, b) # effect size
            value = "{:.3f}\textsuperscript{{{}}}".format(g, self._get_asteriks(p))
            if np.mean(a) > np.mean(b):
                value = "\textbf{{{}}}".format(value)
            return value
        data = {no_features: get_value(aucs_filtered[condition_better][no_features], aucs_filtered[condition_other][no_features]) for no_features in self.feature_range}
        series = pd.Series(data, name=dataset_name)
        return series

class EffectSizeSingle:

    @staticmethod
    def _get_asteriks(p):
            asteriks = "+"
            if p <= 0.05:
                asteriks = "*"
            if p <= 0.01:
                asteriks = "**"
            if p <= 0.001:
                asteriks = "***"
            return asteriks

    def get_value(self, a, b):
        """
        Hedges g + t-test welch
        :param a:
        :param b:
        :return:
        """
        t, p = scipy.stats.ttest_ind(a, b, equal_var=False)
        g = hedges_g(a, b) # effect size
        value = "{:.3f}{}".format(g, self._get_asteriks(p))
        return value

    def get_correlation(self, a, b):
        """
        Spearman
        :param a:
        :param b:
        :return:
        """
        print('--')
        print(len(a))
        print(len(b))
        if len(a) != len(b):
            m = min(len(a), len(b))
            print(m)
            a = np.random.choice(a, m, replace=False)
            b = np.random.choice(b, m, replace=False)
        corr, p = spearmanr(a, b)
        value = "{:.3f}{}".format(corr, self._get_asteriks(p))
        return value


