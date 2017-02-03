from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd
import sys

from CSFSEvaluator import CSFSEvaluator
from application.CSFSFeatureRecommender import Recommender
from infoformulas_listcomp import IG_from_series
from infoformulas_listcomp import _H


class AUCCalculator:
    """
    EvaluationRankingEvaluator. Calculates the AUC for a budget or a budget range given an ordered list of features with cost and IG
    For each crowd answer, one EREvaluator is needed
    """
    def __init__(self, df_features_ranked, df_cleaned_bin, target):
        self.df_features_ranked = df_features_ranked
        self.df_cleaned_bin = df_cleaned_bin
        self.target = target
        self.auc_evaluator = CSFSEvaluator(df_cleaned_bin, target)

class AucForBudgetCalculator(AUCCalculator):
    """
    EvaluationRankingEvaluator. Calculates the AUC for a budget or a budget range given an ordered list of features with cost and IG
    For each crowd answer, one EREvaluator is needed
    """

    def get_selected_index(self, budget):
        index = list()
        costs = 0
        for i, row in self.df_features_ranked.iterrows():
            if costs + row.Cost <= budget:
                costs += row.Cost
                index.append(i)
            else:
                break
        return index


    def get_auc_for_budget(self, budget):
        index_selected = self.get_selected_index(budget)
        if len(index_selected) == 0:
            print('Budget too low. No features selected')
            return 0

        features = list(self.df_features_ranked['Feature'].loc[index_selected])
        auc = self.auc_evaluator.evaluate_features(features)
        return auc

    def get_auc_for_budget_range(self, budget_range):
        result = {budget: self.get_auc_for_budget(budget) for budget in budget_range}
        result = pd.DataFrame.from_dict(result, orient='index')
        result.transpose()
        result.columns = ['auc']
        return result.sort_index()


class AUCForOrderedFeaturesCalculator(AUCCalculator):

    def get_auc_for_nofeatures(self, n):
        features = list(self.df_features_ranked['Feature'].loc[:n-1]) # index starting with 0 and includes n
        auc = self.auc_evaluator.evaluate_features(features)
        return auc

    def get_auc_for_nofeatures_range(self, n_range):
        result = {n: self.get_auc_for_nofeatures(n) for n in n_range}
        result = pd.DataFrame.from_dict(result, orient='index')
        result.transpose()
        result.columns = ['auc']
        return result.sort_index()


class Evaluation(metaclass=ABCMeta):
    def __init__(self, path_cost_ig, path_cleaned_bin, target):
        df = pd.read_csv(path_cost_ig, index_col=0)
        data = {'feature': list(df.index), 'cost': list(df['Cost']), 'IG': list(df['IG median'])}
        self.df_cost_ig = pd.DataFrame(data)
        self.count_features_all = len(self.df_cost_ig['feature'])
        self.recommender = Recommender(self.df_cost_ig)
        df_data = pd.read_csv(path_cleaned_bin)
        self.evaluator = CSFSEvaluator(df_data, target)

    @abstractmethod
    def get_auc_for_budget(self, budget):
        return

    def get_auc_for_budget_range(self, budget_range):
        """

        :param budget_range: list(int)
        :return: pd.DataFrame with columns: bestvalue, AUC and no_features
        """
        result = {budget: self.get_auc_for_budget(budget) for budget in budget_range}
        result = pd.DataFrame(result).transpose()
        result.columns = ['bestvalue', 'auc', 'features', 'count_features_ratio']
        return result


class TestEvaluation(Evaluation):
    def __init__(self, path_cost_ig, path_cleaned_bin, target):
        df = pd.read_csv(path_cost_ig, index_col=0)
        data = {'feature': list(df.index), 'cost': list(df['Cost']), 'IG': list(df['IG median'])}
        self.df_cost_ig = pd.DataFrame(data)
        self.count_features_all = len(self.df_cost_ig['feature'])
        self.recommender = Recommender(self.df_cost_ig)

        df_data = pd.read_csv(path_cleaned_bin)
        self.evaluator = CSFSEvaluator(df_data, target)

    def get_auc_for_budget(self, budget):
        bestvalue, features = self.recommender.recommend_for_budget(budget)
        auc = self.evaluator.evaluate_features(features)
        return bestvalue, auc, features, len(features)/self.count_features_all

    def get_auc_for_nofeatures(self, features):
        auc = self.evaluator.evaluate_features(features)
        return np.nan, auc, features, len(features)/self.count_features_all

    def get_auc_for_nofeatures_range(self, features_range):
        self.df_cost_ig = self.df_cost_ig.sort_values('IG', ascending=False)
        features = list(self.df_cost_ig['feature'])
        result = {i: self.get_auc_for_nofeatures(features[:i]) for i in features_range}
        result = pd.DataFrame(result).transpose()
        result.columns = ['bestvalue', 'auc', 'features', 'count_features_ratio']
        return result



class FinalEvaluationCSFSCondition:
    """
    Evaluation for condition 4 (CSFS)
    """

    def __init__(self, df_cleaned_bin, target, dataset_name, df_answers_grouped, bootstrap_n=12, repetitions=100):
        self.df_cleaned_bin = df_cleaned_bin
        self.target = target
        self.dataset_name = dataset_name
        self.df_answers_grouped = df_answers_grouped
        self.bootstrap_n = bootstrap_n
        self.repetitions = repetitions

    def evaluate(self, budget_range):
        def bootstrap_row(row):
            p = list(row['p'])
            pf0 = list(row['p|f=0'])
            pf1 = list(row['p|f=1'])
            row['p'] = np.random.choice(p, replace=True, size=self.bootstrap_n)
            row['p|f=0'] = np.random.choice(pf0, replace=True, size=self.bootstrap_n)
            row['p|f=1'] = np.random.choice(pf1, replace=True, size=self.bootstrap_n)
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
        for i in range(self.repetitions): # number of iterations for bootstrapping -> is number of aucs calculated
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
        return result


def test():
    path_cost_ig = 'conditions/test/olympia.csv'
    path_cleaned_bin = '../datasets/olympia/cleaned/experiment2-4_all/olympic_allyears_plus_bin.csv'
    target = 'medals'

    test_evaluation = TestEvaluation(path_cost_ig, path_cleaned_bin, target)

    budget_range = [10, 20, 30, 40, 50]
    test_evaluation.get_auc_for_budget_range(budget_range)


if __name__ == '__main__':
    test()