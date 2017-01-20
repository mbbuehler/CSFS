import numpy as np
import pandas as pd
import statsmodels.stats.weightstats as ssw

from application.CSFSConditionEvaluation import AUCForOrderedFeaturesCalculator, FinalEvaluationCSFSCondition
from application.EvaluationRanking import ERParser, ERFilterer, ERCondition, ERNofeaturesEvaluator


class FinalEvaluationCombiner:

    def __init__(self, df_evaluation_result, df_evaluation_base, df_cleaned_bin, target, dataset_name, df_answers_grouped, bootstrap_n=12, repetitions=100, ):
        self.df_evaluation_result = df_evaluation_result
        self.df_evaluation_base = df_evaluation_base
        self.df_cleaned_bin = df_cleaned_bin
        self.parser = ERParser(df_evaluation_base)
        self.target = target
        self.dataset_name = dataset_name
        self.df_answers_grouped = df_answers_grouped
        self.bootstrap_n = bootstrap_n
        self.repetitions = repetitions

    def combine(self, feature_range):
        df_combined = pd.DataFrame(columns=['number_of_features', 'dataset', 'ranking_strategy', 'user_id', 'AUC', 'AUC_95_CI_low', 'AUC_95_CI_high'])

        # ranking conditions
        for condition in [1, 2, 3]:
            df_combined = df_combined.append(self.get_ranking_condition(feature_range, condition))

        # csfs condition
        df_combined = df_combined.append(self.get_csfs_condition(feature_range))

        # add CI
        df_combined = self.update_with_ci(df_combined, feature_range)
        return df_combined

    def get_csfs_condition(self, feature_range, condition=4):
        df = pd.DataFrame(columns=['number_of_features', 'dataset', 'ranking_strategy', 'user_id', 'AUC', 'AUC_95_CI_low', 'AUC_95_CI_high'])
        result = FinalEvaluationCSFSCondition(self.df_cleaned_bin, self.target, self.dataset_name, self.df_answers_grouped, self.bootstrap_n, self.repetitions).evaluate(feature_range)
        # {1: [0.62143308936477815, 0.62143308936477815, 0.62143308936477815, 0.62143308936477815,...
        for nofeature in result:
            for i, auc in enumerate(result[nofeature]):
                df = df.append({
                    'number_of_features': nofeature,
                    'dataset': self.dataset_name,
                    'ranking_strategy': ERCondition.get_string(condition),
                    'user_id': i,
                    'AUC': auc,
                    'AUC_95_CI_low': 0,
                    'AUC_95_CI_high': 0,
                    }, ignore_index=True)
        return df

    def get_ranking_condition(self, feature_range, condition):
        df = pd.DataFrame(columns=['number_of_features', 'dataset', 'ranking_strategy', 'user_id', 'AUC', 'AUC_95_CI_low', 'AUC_95_CI_high'])
        filterer = ERFilterer(self.dataset_name, condition, remove_test=True)
        df_result_filtered = filterer.get_filtered_result(self.df_evaluation_result)
        for i,row in df_result_filtered.iterrows():
            df_features_ranked = self.parser.get_ordered_features(row.token)
            evaluator = AUCForOrderedFeaturesCalculator(df_features_ranked, self.df_cleaned_bin, self.target)
            df_aucs = evaluator.get_auc_for_nofeatures_range(feature_range) # df with one col: AUC and index= cost
            for j, costrow in df_aucs.iterrows():
                df = df.append({
                    'number_of_features': j,
                    'dataset': self.dataset_name,
                    'ranking_strategy': ERCondition.get_string(condition),
                    'user_id': row['name'],
                    'AUC': costrow.auc,
                    'AUC_95_CI_low': 0,
                    'AUC_95_CI_high': 0,
                }, ignore_index=True)

        # calculate CI

        return df

    def update_with_ci(self, df, feature_range):
        for nofeature in feature_range:
            indices = df['number_of_features'] == nofeature
            aucs = df.loc[indices, 'AUC']
            ci = ssw.DescrStatsW(aucs).tconfint_mean()
            df.loc[indices, 'AUC_95_CI_low'] = ci[0]
            df.loc[indices, 'AUC_95_CI_high'] = ci[1]
        return df