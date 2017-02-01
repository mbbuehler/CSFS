from collections import Counter

import numpy as np
import pandas as pd
from tabulate import tabulate

from CSFSCrowdCleaner import CSFSCrowdCleaner, CSFSCrowdAggregator, CSFSCrowdAnalyser
from CSFSDataPreparator import DataPreparator
from abstract_experiment import AbstractExperiment
from application.EvaluationRanking import ERFilterer, ERCondition, ERParser


class ExperimentStudent(AbstractExperiment):

    def __init__(self, dataset_name, experiment_number, experiment_name):
        super().__init__(dataset_name, experiment_number, experiment_name)
        self.feature_range = range(1, 16)

        self.path_raw = '{}raw/{}/student-por.csv'.format(self.base_path, experiment_name)
        self.path_cleaned = '{}cleaned/{}/student-por_clean.csv'.format(self.base_path, experiment_name)
        self.path_bin = '{}cleaned/{}/student-por_clean_bin.csv'.format(self.base_path, experiment_name)
        self.path_autocorrelation = '{}cleaned/{}/student-por_bin_autocorrelation.csv'.format(self.base_path, experiment_name)
        self.path_meta = '{}cleaned/{}/student-por_clean_bin_meta.csv'.format(self.base_path, experiment_name)
        self.path_answers_raw = '{}results/{}/answers_raw.xlsx'.format(self.base_path, experiment_name)
        self.path_answers_clean = '{}results/{}/answers_clean.csv'.format(self.base_path, experiment_name)
        self.path_answers_clean_grouped = '{}results/{}/answers_clean_grouped.pickle'.format(self.base_path, experiment_name)
        self.path_answers_plots = '{}results/{}/visualisations/{}_histograms_answers.html'.format(self.base_path, experiment_name, self.dataset_name)
        self.path_answers_aggregated = '{}results/{}/answers_aggregated.csv'.format(self.base_path, experiment_name)
        self.path_answers_metadata = '{}results/{}/answers_metadata.csv'.format(self.base_path, experiment_name)
        self.path_no_answers_vs_auc = '{}results/{}/answers_vs_auc.pickle'.format(self.base_path, experiment_name)
        self.path_answers_delta = '{}results/{}/answers_delta.pickle'.format(self.base_path, experiment_name)

        self.path_csfs_auc = '{}results/{}/csfs_auc.csv'.format(self.base_path, experiment_name)
        self.path_csfs_std = '{}results/{}/csfs_std.csv'.format(self.base_path, experiment_name)
        self.path_questions = '{}questions/{}/questions_high-school.csv'.format(self.base_path, experiment_name) # experiment2 for experiment3
        self.path_flock_result = '{}results/{}/flock_auc.csv'.format(self.base_path, experiment_name)

        self.path_cost_ig_test = 'application/conditions/test/student.csv'
        self.path_cost_ig_expert = 'application/conditions/expert/student.csv'
        self.path_budget_evaluation_cost = '{}evaluation/budget_evaluation_cost.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_nofeatures = '{}evaluation/budget_evaluation_nofeatures.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_cost_rawaucs = '{}evaluation/budget_evaluation_cost_rawaucs.pickle'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_nofeatures_rawaucs = '{}evaluation/budget_evaluation_nofeatures_rawaucs.pickle'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_result_domain = '{}evaluation/experts_domain/result_domain.csv'.format(self.base_path)
        self.path_cost_ig_base = '{}evaluation/student_base.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_base = '{}evaluation/base.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_result = 'final_evaluation/conditions1-3_result.csv'

        self.path_final_evaluation_aucs = '{}evaluation/final_evaluation_aucs.pickle'.format(self.base_path)
        self.path_final_evaluation_aggregated = '{}evaluation/final_evaluation_aggregated.pickle'.format(self.base_path)
        self.path_final_evaluation_combined = '{}evaluation/final_evaluation_combined.csv'.format(self.base_path)
        self.path_auc_plots = '{}evaluation/visualisation/{}_histograms_aucs.html'.format(self.base_path, self.dataset_name)



        self.path_descriptions_domain = '{}evaluation/experts_domain/student_descriptions_domain.csv'.format(self.base_path)
        self.target = 'G3'


    def preprocess_raw(self):
        """
        Selects only interesting features, fills gaps
        outputs a csv into "cleaned" folder
        :return:
        """
        df_raw = pd.read_csv(self.path_raw, quotechar='"', delimiter=';')

        features_to_remove = ['G1', 'G2']
        preparator = DataPreparator()

        # only take subset we have questions for
        df_raw = preparator.drop_columns(df_raw, features_to_remove)
        df_raw.to_csv(self.path_cleaned, index=False)

    def bin_binarise(self):
        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """
        df = pd.read_csv(self.path_cleaned)
        target_median = np.median(df[self.target])

        df[self.target] = df[self.target].apply(lambda x: 1 if x >= target_median else 0) # 1:"belongs to the better one" 0: "belongs to the lower half or middle"

        preparator = DataPreparator()
        df = preparator.prepare(df, columns_to_ignore=[self.target])
        df.to_csv(self.path_bin, index=False)

    # def evaluate_crowd_all_answers(self):
    #     """
    #     Aggregates crowd answers and evaluates for all crowd answers
    #     :return:
    #     """
    #     df_clean = CSFSCrowdCleaner(self.path_questions, self.path_answers_raw, self.target).clean()
    #     df_clean.to_csv(self.path_answers_clean, index=True)
    #     print(df_clean)
    #     exit()
    #
    #     df_aggregated = CSFSCrowdAggregator(df_clean, target=self.target, mode=CSFSCrowdAggregator.Mode.EXTENDED, ).aggregate()
    #     df_aggregated.to_csv(self.path_answers_aggregated, index=True)
    #
    #     df_combined = CSFSCrowdAnalyser().get_combined_df(self.path_answers_aggregated, self.path_meta)
    #     df_combined.to_csv(self.path_answers_metadata, index=True)

    def domain_evaluation(self):
        """
        saves df with columns: rank (e.g. first rank counted paid==yes once), index: features, values: counts
                                1    2    3    4    5    6    7    8    9    10    11    12    13    14    15
---------------------  ---  ---  ---  ---  ---  ---  ---  ---  ---  ----  ----  ----  ----  ----  ----
Pstatus==T               0    0    0    0    0    1    1    2    2     0     0     1     0     0     0
failures_(-0.003, 1]     1    2    0    1    1    0    0    1    0     0     0     0     1     0     0
paid==yes                1    1    0    0    1    2    0    0    0     0     1     0     0     1     0
        :return:
        """
        df = pd.read_csv(self.path_budget_evaluation_result, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        df = ERFilterer(self.dataset_name, ERCondition.DOMAIN).get_filtered_result(df)
        tokens = list(df['token'])
        df_evaluation_base = pd.read_csv(self.path_budget_evaluation_base)
        parser = ERParser(df_evaluation_base)

        feature_lists = list()
        for token in tokens:
            features = list(parser.get_ordered_features(token)['Feature'])
            feature_lists.append(features)
        features_all = set(feature_lists[0])
        n = len(features_all) # number of features
        m = len(feature_lists) # number of responses
        data = {i+1:[feature_lists[l][i] for l in range(m)] for i in range(n)}
        df_counted = pd.DataFrame(index=features_all, columns=[i for i in data])
        for i in data:
            counted = Counter(data[i])
            for f in counted:
                df_counted[i].loc[f] = counted[f]
        df_counted = df_counted.fillna(0)

        # only for joining feature names with readable names
        # df_descriptions = pd.read_csv(self.path_descriptions_domain, header=0)
        # df_joined = pd.merge(df_descriptions, df_counted, left_on='Feature', right_index=True)
        # df_joined = df_joined.drop(['No', 'Feature', 'Cost', 'Description'], axis=1)
        # df_joined = df_joined.set_index('Name')

        df_counted.to_csv(self.path_budget_evaluation_result_domain)






if __name__ == '__main__':
    experiment = ExperimentStudent('student', 2, 'experiment2_por')
    fake_features={'G3': 0.5}

    auto_open_plots = False
    # experiment.set_up_basic_folder_structure()
    # experiment.set_up_experiment_folder_structure('experiment2_por')
    # experiment.preprocess_raw()
    # experiment.bin_binarise()
    # experiment.get_metadata()
    # experiment.evaluate_crowd_all_answers(fake_features=fake_features)
     # experiment.drop_analysis(N_Features, n_samples)
    # experiment.evaluate_flock(N_Features, n_samples, range(3, 350, 1))
    # experiment.evaluate_csfs_auc(fake_features={'G3': 0.5})
    # experiment.drop_evaluation(N_Features, n_samples)
    # budget_range = range(10, 180, 10)

    # experiment.evaluate_budget(budget_range)
    # df_budget_evaluation = pd.read_csv(experiment.path_budget_evaluation, index_col=0, header=[0, 1])
    # experiment.get_figure_budget_evaluation(df_budget_evaluation)
    # experiment.evaluate_ranking_cost(budget_range)
    # experiment.evaluate_ranking_nofeatures(no_features)
        #
    # experiment.evaluate_csfs_auc()
    # experiment.domain_evaluation()
    # experiment.autocorrelation()
    # experiment.final_evaluation(feature_range, bootstrap_n=12, repetitions=20)
    # experiment.final_evaluation_visualisation(feature_range)
    # experiment.crowd_answers_plot(auto_open=auto_open_plots)
    # experiment.final_evaluation_combine(feature_range, bootstrap_n=12, repetitions=20)
    # experiment.crowd_auc_plot(auto_open=auto_open_plots)
    # experiment.statistical_comparison(feature_range)
    # experiment.evaluate_no_answers()
    # experiment.evaluate_no_answers_get_fig(feature_range)
    # experiment.evaluate_answers_delta()
    experiment.evaluate_answers_delta_plot(auto_open=True)
    # experiment.humans_vs_actual_auc()
    # experiment.humans_vs_actual_auc_plot()

