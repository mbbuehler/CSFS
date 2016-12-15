from abc import abstractmethod, ABCMeta

import pandas as pd

from CSFSEvaluator import CSFSEvaluator
from application.CSFSFeatureRecommender import Recommender


class AucForBudgetCalculator:
    """
    EvaluationRankingEvaluator. Calculates the AUC for a budget or a budget range given an ordered list of features with cost and IG
    For each crowd answer, one EREvaluator is needed
    """
    def __init__(self, df_features_ranked, df_cleaned_bin, target):
        self.df_features_ranked = df_features_ranked
        self.df_cleaned_bin = df_cleaned_bin
        self.target = target
        self.auc_evaluator = CSFSEvaluator(df_cleaned_bin, target)

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
        result.columns = ['AUC']
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
        result.columns = ['bestvalue', 'AUC', 'features', 'count_features_ratio']
        return result


class RankingEvaluation(Evaluation):
    def __init__(self, path_cost_ig, path_cleaned_bin, target):
        Evaluation.__init__(self, path_cost_ig, path_cleaned_bin, target)

    def get_selected_index(self, budget):
        index = list()
        costs = 0
        for i, row in self.df_cost_ig.iterrows():
            if costs + row.cost <= budget:
                costs += row.cost
                index.append(i)
            else:
                break
        return index


    def get_auc_for_budget(self, budget):
        index_selected = self.get_selected_index(budget)
        features = list(self.df_cost_ig['feature'].loc[index_selected])
        bestvalue = sum(self.df_cost_ig['IG'].loc[index_selected])
        auc = self.evaluator.evaluate_features(features)
        return bestvalue, auc, features, len(features)/self.count_features_all


class TestEvaluation:

    def __init__(self, path_cost_ig, path_cleaned_bin, target):
        df = pd.read_csv(path_cost_ig, index_col=0)
        data = {'feature': list(df.index), 'cost': list(df['Cost']), 'IG': list(df['IG median'])}
        df_cost_ig = pd.DataFrame(data)
        self.count_features_all = len(df_cost_ig['feature'])
        self.recommender = Recommender(df_cost_ig)

        df_data = pd.read_csv(path_cleaned_bin)
        self.evaluator = CSFSEvaluator(df_data, target)

    def get_auc_for_budget(self, budget):
        bestvalue, features = self.recommender.recommend_for_budget(budget)
        auc = self.evaluator.evaluate_features(features)
        return bestvalue, auc, features, len(features)/self.count_features_all



def test():
    path_cost_ig = 'conditions/test/olympia.csv'
    path_cleaned_bin = '../datasets/olympia/cleaned/experiment2-4_all/olympic_allyears_plus_bin.csv'
    target = 'medals'

    test_evaluation = TestEvaluation(path_cost_ig, path_cleaned_bin, target)

    budget_range = [10, 20, 30, 40, 50]
    test_evaluation.get_auc_for_budget_range(budget_range)


if __name__ == '__main__':
    test()