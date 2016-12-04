import pandas as pd

from CSFSEvaluator import CSFSEvaluator
from application.CSFSFeatureRecommender import Recommender


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

    def get_auc_for_budget_range(self, budget_range):
        """

        :param budget_range: list(int)
        :return: pd.DataFrame with columns: bestvalue, AUC and no_features
        """
        result = {budget: self.get_auc_for_budget(budget) for budget in budget_range}
        result = pd.DataFrame(result).transpose()
        result.columns = ['bestvalue', 'AUC', 'features', 'count_features_ratio']
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