import re
from sys import stdout
from time import time

import numpy as np
import pandas as pd
import scipy.stats as st
import sys
from joblib import Parallel, delayed
from tabulate import tabulate

from application.CSFSConditionEvaluation import AucForBudgetCalculator, AUCForOrderedFeaturesCalculator
from infoformulas_listcomp import _H, IG_from_series


class ERCondition:
    LAYPERSON = 1 # AMT Turkers
    DOMAIN = 2 # e.g. Teachers
    EXPERT = 3 # Upwork
    CSFS = 4
    RANDOM = 5
    ACTUAL = 6
    BEST = 7
    WORST = 8
    HUMAN = 9

    NAMES_SHORT = {
        LAYPERSON: 'lay',
        DOMAIN: 'domain',
        EXPERT: 'experts',
        CSFS: 'csfs',
        RANDOM: 'random',
    }
    NAMES = {
        LAYPERSON: 'lay',
        DOMAIN: 'domain expert',
        EXPERT: 'data scientist',
        CSFS: 'CSFS',
        RANDOM: 'random',
        ACTUAL: 'actual',
        HUMAN: 'human experts'
    }

    PAPER_NAMES = {
        LAYPERSON: 'Laypeople',
        DOMAIN: 'Domain Experts',
        EXPERT: 'Data Scientists',
        CSFS: 'KrowDD',
        RANDOM: 'Random',
        ACTUAL: 'actual',
        BEST: 'Best',
        WORST: 'Worst',
        HUMAN: 'Human Experts'
    }

    @staticmethod
    def get_all():
        return [ERCondition.LAYPERSON, ERCondition.DOMAIN, ERCondition.EXPERT, ERCondition.CSFS, ERCondition.RANDOM, ERCondition.ACTUAL]

    @staticmethod
    def get_string(condition):
        if condition in ERCondition.NAMES:
            return ERCondition.NAMES[condition]
        return 'n.a.'

    @staticmethod
    def get_string_paper(condition):
        if condition in ERCondition.PAPER_NAMES:
            return ERCondition.PAPER_NAMES[condition]
        print(condition)
        return 'n.a.'

    @staticmethod
    def get_string_short(condition):
        if condition in ERCondition.NAMES_SHORT:
            return ERCondition.NAMES_SHORT[condition]
        return 'n.a.'



    @classmethod
    def get_string_identifier(cls, condition):
        name = cls.NAMES[condition]
        name = name.lower()
        return "_".join([s for s in name.split(' ')])


class ERFilterer:

    def __init__(self, dataset_name, condition, remove_test=False):
        """

        :param dataset_name:
        :param condition:
        :param remove_test: whether to remove rows with 'test' in the name field
        :return:
        """
        self.dataset_name = dataset_name
        self.condition = condition
        self.remove_test = remove_test

    def get_filtered_result(self, df_evaluation_result):
        """
        Removes all conditions but condition from df_result
        :param condition: int
        :return: df
        """
        df_result_filtered = df_evaluation_result[(df_evaluation_result['condition'] == self.condition) & (df_evaluation_result['dataset_name'] == self.dataset_name)]
        if self.remove_test:
            index_keep = df_result_filtered['name'].str.lower() != 'test'
            df_result_filtered = df_result_filtered[index_keep]
        print('Filterer removed {} rows from {} rows'.format(len(df_evaluation_result) - len(df_result_filtered), len(df_evaluation_result)))
        if len(df_result_filtered) == 0:
            # no answers available
            return pd.DataFrame()
        return df_result_filtered

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

    def __init__(self, df_evaluation_result, df_evaluation_base, df_cleaned_bin, target, dataset_name, df_answers_grouped, df_actual_metadata, bootstrap_n=12, repetitions=100, replace=False):
        self.df_evaluation_result = df_evaluation_result
        self.df_evaluation_base = df_evaluation_base
        self.df_cleaned_bin = df_cleaned_bin
        self.df_actual_metadata = df_actual_metadata
        self.parser = ERParser(df_evaluation_base)
        self.target = target
        self.dataset_name = dataset_name
        self.df_answers_grouped = df_answers_grouped
        self.bootstrap_n = bootstrap_n
        self.repetitions = repetitions
        self.replace = replace # if true -> bootstrapping, else sampling without replacement

    def evaluate(self, budget_range, condition):
        pass

    def _get_filtered_result(self, condition):
        """
        Removes all conditions but condition from df_result
        :param condition: int
        :return: df
        """
        filterer = ERFilterer(self.dataset_name, condition, remove_test=True)
        df_result_filtered = filterer.get_filtered_result(self.df_evaluation_result)
        return df_result_filtered

    def _get_df_evaluated(self, df):
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

    def evaluate_all_to_dict(self, budget_range):
        raw_data = Parallel(n_jobs=4)(delayed(self.evaluate)(budget_range, condition) for condition in ERCondition.get_all())
        # [{1: {1: [0.580804915514593, ...
        result = dict()
        for i in range(len(raw_data)):
            result.update(raw_data[i])
        # print(result)
        # raw_data = {condition:() }
        # raw_data = dict()
        # for condition in ERCondition.get_all():
        #     print("> Evaluating condition {}".format(condition))
        #     raw = self.evaluate(budget_range, condition)
        #     raw_data[condition] = raw
        #     # raw_data is dict: {CONDITION: {NOFEATURES: [AUCS]}}
        return result

    def evaluate_all(self, budget_range):
        """
        returns {condition: DF, } e.g. {2: Df}
        :param budget_range:
        :return:dict
        """
        raw_data, evaluated = self.evaluate_all_to_dict(budget_range)
        df_raw = pd.DataFrame(raw_data, index=budget_range)
        return df_raw, evaluated



class ERCostEvaluator(EREvaluator):

    def evaluate(self, budget_range, condition):
        """

        :param budget_range:
        :param condition:
        :return: df_budget_aucs (raw df with columns = cost, index all 'auc') and df_evaluated (df with CI usw.)
        """
        df_result_filtered = self._get_filtered_result(condition)
        list_budget_aucs = [self._get_aucs(row, budget_range) for i,row in df_result_filtered.iterrows()]
        df_budget_aucs = pd.concat(list_budget_aucs, axis='columns').transpose() # df with columns index= x times 'AUC' and columns=cost
        df_evaluated = self._get_df_evaluated(df_budget_aucs)
        return df_budget_aucs, df_evaluated



    def _get_aucs(self, row, budget_range):
        token = row.token
        df_features_ranked = self.parser.get_ordered_features(token)
        evaluator = AucForBudgetCalculator(df_features_ranked, self.df_cleaned_bin, self.target)
        df_aucs = evaluator.get_auc_for_budget_range(budget_range) # df with one col: AUC and index= cost
        return df_aucs



class ERNofeaturesEvaluator(EREvaluator):

    def evaluate(self, budget_range, condition):
        sys.stdout.write("> Condition {} started".format(condition))
        if condition in [ERCondition.LAYPERSON, ERCondition.DOMAIN, ERCondition.EXPERT]: # ranking conditions
            df_result_filtered = self._get_filtered_result(condition)
            list_nofeatures_aucs = [self._get_aucs(row, budget_range) for i,row in df_result_filtered.iterrows()] # list of dfs with index=nofeature and one column 'auc'
            result = {int(nofeature): list() for nofeature in list_nofeatures_aucs[0].index}
            for nofeature in result:
                result[nofeature] = [float(df.loc[nofeature]) for df in list_nofeatures_aucs]

        elif condition == ERCondition.CSFS: #csfs condition
            def bootstrap_row(row):
                p = list(row['p'])
                pf0 = list(row['p|f=0'])
                pf1 = list(row['p|f=1'])
                row['p'] = np.random.choice(p, replace=self.replace, size=self.bootstrap_n)
                row['p|f=0'] = np.random.choice(pf0, replace=self.replace, size=self.bootstrap_n)
                row['p|f=1'] = np.random.choice(pf1, replace=self.replace, size=self.bootstrap_n)
                return row
            def aggregate(row):
                row['p'] = np.median(row['p'])
                row['p|f=0'] = np.median(row['p|f=0'])
                row['p|f=1'] = np.median(row['p|f=1'])
                return row
            def calc_ig(row, p_target):
                h_x = _H([p_target, 1-p_target])
                row['IG'] = IG_from_series(row, h_x=h_x, identifier='p')
                return row

            result = {nofeatures: list() for nofeatures in budget_range}
            p_target = self.df_answers_grouped['p'].loc[self.target][0]
            df_answers_tmp = self.df_answers_grouped.drop(self.target) # need to drop target
            for i in range(self.repetitions): # is number of aucs calculated
                sys.stdout.write(str(i)+" ")
                # bootstrap answers
                df_answers_bootstrapped = df_answers_tmp.copy().apply(bootstrap_row, axis='columns')
                df_aggregated = df_answers_bootstrapped.apply(aggregate, axis='columns')
                df_aggregated = df_aggregated.apply(calc_ig, axis='columns', p_target=p_target)
                df_ordered = df_aggregated.sort_values('IG', ascending=False)
                # reset index
                df_ordered['Feature'] = df_ordered.index
                df_ordered = df_ordered.reset_index()

                evaluator = AUCForOrderedFeaturesCalculator(df_ordered, self.df_cleaned_bin, self.target)
                df_aucs = evaluator.get_auc_for_nofeatures_range(budget_range) # df with one col: AUC and index= cost
                for nofeature in df_aucs.index:
                    result[nofeature].append(df_aucs.loc[nofeature]['auc'])

        elif condition == ERCondition.RANDOM: # random
            features = list(self.df_answers_grouped.drop(self.target).index)
            df_ordered = pd.DataFrame({'Feature': features})
            result = {nofeatures: list() for nofeatures in budget_range}
            for i in range(19):
                sys.stdout.write("Random Repetition {}".format(i))
                df_shuffled = df_ordered.sample(frac=1).reset_index(drop=True)
                # print(df_shuffled)
                evaluator = AUCForOrderedFeaturesCalculator(df_shuffled, self.df_cleaned_bin, self.target)
                df_aucs = evaluator.get_auc_for_nofeatures_range(budget_range) # df with one col: AUC and index= cost
                for nofeature in df_aucs.index:
                    result[nofeature].append(df_aucs.loc[nofeature]['auc'])

        elif condition == ERCondition.ACTUAL: # using IGs from actual values
            result = {nofeatures: -1 for nofeatures in budget_range}
            df = self.df_actual_metadata.drop(self.target)
            df_ordered = df.sort_values('IG', ascending=False)
            # reset index
            df_ordered['Feature'] = df_ordered.index
            df_ordered = df_ordered.reset_index()
            evaluator = AUCForOrderedFeaturesCalculator(df_ordered, self.df_cleaned_bin, self.target)
            df_aucs = evaluator.get_auc_for_nofeatures_range(budget_range) # df with one col: AUC and index= cost
            for nofeature in df_aucs.index:
                result[nofeature] = [df_aucs.loc[nofeature]['auc']] # list only used to keep same format as other conditions
        print('> condition {} done'.format(condition))
        return {condition: result}

    def _get_aucs(self, row, budget_range):
        token = row.token
        df_features_ranked = self.parser.get_ordered_features(token)
        evaluator = AUCForOrderedFeaturesCalculator(df_features_ranked, self.df_cleaned_bin, self.target)
        df_aucs = evaluator.get_auc_for_nofeatures_range(budget_range) # df with one col: AUC and index= cost
        return df_aucs

def test():
    token = '0:13,1:14,2:1,3:3,4:4,5:7,6:11,7:5,8:12,9:15,10:6,11:2,12:8,13:9,14:10|e7cf0fccca7858d47a96c82837e6d439'
    path = '../datasets/student/evaluation/student_base.csv'
    ERParser(path).parse(token)

    features = ERParser(path).get_ordered_features(token)
    print(features)

if __name__ == '__main__':
    test()