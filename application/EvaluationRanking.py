import re

import numpy as np
import pandas as pd
import scipy.stats as st

from application.CSFSConditionEvaluation import AucForBudgetCalculator

class ERCondition:
    LAYPERSON = 1 # AMT Turkers
    DOMAIN = 2 # e.g. Teachers
    EXPERT = 3 # Upwork
    TEST = 4

    @staticmethod
    def get_all():
        return [ERCondition.LAYPERSON, ERCondition.DOMAIN, ERCondition.EXPERT]


class ERParser:
    def __init__(self, df_evaluation_base):
        """

        :param df_evaluation_base: df_cost_ig
        :return:
        """
        self.df_evaluation_base = df_evaluation_base

    def _extract_index_value(self, ranking):
        indeces = list()
        values = list()
        for element in ranking.split(','):
            match = re.search(r'(\d+):(\d+)', element)
            index = match.group(1)
            value = match.group(2)
            indeces.append(index)
            values.append(value)
        # need integers
        return dict(index=np.array(indeces, dtype='int'), No=np.array(values, dtype='int'))

    def parse(self, token):
        ranking = re.search(r'^(.*)\|.+$', token).group(1)
        df_ranking = pd.DataFrame(self._extract_index_value(ranking))
        df_merged = pd.merge(df_ranking, self.df_evaluation_base, on='No')
        df_merged.index = list(df_merged['index'])
        df_merged = df_merged.drop('index', axis=1)
        return df_merged

    def get_ordered_features(self, token):
        """
        Extracts the ordered list of features from token
        :param token: str e.g. '0:13,1:14,2:1,3:3,4:4,5:7,6:11,7:5,8:12,9:15,10:6,11:2,12:8,13:9,14:10|e7cf0fccca7858d47a96c82837e6d439'
        :return: list(str)
        """
        df_merged = self.parse(token)
        return df_merged


class EREvaluator:
    """

         ci_95_hi  ci_95_low      mean       std
10        NaN        NaN  0.000000  0.000000
20   0.781988   0.135474  0.458731  0.232857
30   0.630208   0.518547  0.574378  0.040217
40   0.635273   0.549133  0.592203  0.031025
50   0.637656   0.553728  0.595692  0.030229
60   0.641157   0.600360  0.620759  0.014694
70   0.639263   0.598034  0.618649  0.0148
    """

    def __init__(self, df_evaluation_result, df_evaluation_base, df_cleaned_bin, target):
        self.df_evaluation_result = df_evaluation_result
        self.df_evaluation_base = df_evaluation_base
        self.df_cleaned_bin = df_cleaned_bin
        self.parser = ERParser(df_evaluation_base)
        self.target = target


    def _get_aucs(self, row, budget_range):
        token = row.token
        df_features_ranked = self.parser.get_ordered_features(token)
        evaluator = AucForBudgetCalculator(df_features_ranked, self.df_cleaned_bin, self.target)
        df_aucs = evaluator.get_auc_for_budget_range(budget_range) # df with one col: AUC and index= cost
        return df_aucs

    def _get_filtered_result(self, condition):
        """
        Removes all conditions but condition from df_result
        :param condition: int
        :return: df
        """
        df_result_filtered = self.df_evaluation_result[self.df_evaluation_result['condition'] == condition]
        if len(df_result_filtered) == 0:
            # no answers available
            return pd.DataFrame()
        return df_result_filtered

    def _get_df_evaluated(self, df, col_val='cost'):
        """
        df with columns index= x times 'AUC' and columns=cost or no_features
        :param df:
        :return:
        """
        mean = np.mean(df)
        std = np.std(df)
        # http://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
        # The underlying assumptions for both are that the sample (array a) was drawn independently from a normal distribution with unknown standard deviation
        ci_intervals = [st.t.interval(0.95, len(df[cost])-1, loc=np.mean(df[cost]), scale=st.sem(df[cost])) for cost in df]
        ci_low = [e[0] for e in ci_intervals]
        ci_high = [e[1] for e in ci_intervals]

        df_evaluated = pd.DataFrame(dict(auc=mean, std=std, ci_lo=ci_low, ci_hi=ci_high))
        return df_evaluated

    def evaluate(self, budget_range, condition):
        df_result_filtered = self._get_filtered_result(condition)

        list_budget_aucs = [self._get_aucs(row, budget_range) for i,row in df_result_filtered.iterrows()]
        df_budget_aucs = pd.concat(list_budget_aucs, axis='columns').transpose() # df with columns index= x times 'AUC' and columns=cost
        df_evaluated = self._get_df_evaluated(df_budget_aucs)
        return df_evaluated

    def evaluate_all(self, budget_range):
        """
        returns {condition: DF, } e.g. {2: Df}
        :param budget_range:
        :return:dict
        """
        data = {condition: self.evaluate(budget_range, condition) for condition in ERCondition.get_all()}

        return data



def test():
    token = '0:13,1:14,2:1,3:3,4:4,5:7,6:11,7:5,8:12,9:15,10:6,11:2,12:8,13:9,14:10|e7cf0fccca7858d47a96c82837e6d439'
    path = '../datasets/student/evaluation/student_base.csv'
    ERParser(path).parse(token)

    features = ERParser(path).get_ordered_features(token)
    print(features)

if __name__ == '__main__':
    test()