import numpy as np
import pandas as pd

from CSFSCrowdCleaner import CSFSCrowdCleaner, CSFSCrowdAggregator, CSFSCrowdAnalyser
from CSFSDataPreparator import DataPreparator
from abstract_experiment import AbstractExperiment


class ExperimentIncome(AbstractExperiment):

    def __init__(self, dataset_name, experiment_number, experiment_name):
        super().__init__(dataset_name, experiment_number, experiment_name)

        self.path_raw = '{}raw/{}/income.csv'.format(self.base_path, experiment_name)
        self.path_cleaned = '{}cleaned/{}/income_clean.csv'.format(self.base_path, experiment_name)
        self.path_bin = '{}cleaned/{}/income_clean_bin.csv'.format(self.base_path, experiment_name)
        self.path_autocorrelation = '{}cleaned/{}/income_bin_autocorrelation.csv'.format(self.base_path, experiment_name)
        self.path_meta = '{}cleaned/{}/income_clean_bin_meta.csv'.format(self.base_path, experiment_name)
        self.path_answers_raw = '{}results/{}/answers_raw3.xlsx'.format(self.base_path, experiment_name)
        self.path_answers_clean = '{}results/{}/answers_clean.csv'.format(self.base_path, experiment_name)
        self.path_answers_clean_grouped = '{}results/{}/answers_clean_grouped.pickle'.format(self.base_path, experiment_name)
        self.path_answers_plots = '{}results/{}/visualisations/'.format(self.base_path, experiment_name)
        self.path_answers_aggregated = '{}results/{}/answers_aggregated.csv'.format(self.base_path, experiment_name)
        self.path_answers_metadata = '{}results/{}/answers_metadata.csv'.format(self.base_path, experiment_name)
        self.path_csfs_auc = '{}results/{}/csfs_auc.csv'.format(self.base_path, experiment_name)
        self.path_csfs_std = '{}results/{}/csfs_std.csv'.format(self.base_path, experiment_name)
        self.path_questions = '{}questions/{}/questions.csv'.format(self.base_path, experiment_name)
        self.path_flock_result = '{}results/{}/flock_auc.csv'.format(self.base_path, experiment_name)
        #
        # self.path_cost_ig_test = 'application/conditions/test/income.csv'
        # self.path_cost_ig_expert = 'application/conditions/expert/income.csv'
        # self.path_budget_evaluation = '{}budget/{}/budget_evaluation.csv'.format(self.base_path, experiment_name)
        # self.path_cost_ig_base = '{}evaluation/income_base.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_base = '{}evaluation/base.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_result = '{}evaluation/result.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_cost = '{}evaluation/budget_evaluation_cost.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_nofeatures = '{}evaluation/budget_evaluation_nofeatures.csv'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_cost_rawaucs = '{}evaluation/budget_evaluation_cost_rawaucs.pickle'.format(self.base_path, experiment_name)
        self.path_budget_evaluation_nofeatures_rawaucs = '{}evaluation/budget_evaluation_nofeatures_rawaucs.pickle'.format(self.base_path, experiment_name)

        self.path_final_evaluation_aucs = '{}evaluation/final_evaluation_aucs.pickle'.format(self.base_path)
        self.path_final_evaluation_aggregated = '{}evaluation/final_evaluation_aggregated.pickle'.format(self.base_path)

        self.target = 'income==>50K'


    def preprocess_raw(self):
        """
        Selects only interesting features, fills gaps
        outputs a csv into "cleaned" folder
        :return:
        """
        df_raw = pd.read_csv(self.path_raw, quotechar='"', delimiter=',')
        df_raw.to_csv(self.path_cleaned, index=False)

    def bin_binarise(self):
        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """
        df = pd.read_csv(self.path_cleaned)
        preparator = DataPreparator()
        df = preparator.prepare(df)
        df.to_csv(self.path_bin, index=False)

if __name__ == '__main__':
    experiment = ExperimentIncome('income', 1, 'experiment1')

    N_Features = range(3, 17, 2)
    n_samples = 100 # number of repetitions to calculate average auc score for samples)
    # experiment.set_up_basic_folder_structure()
    # experiment.set_up_experiment_folder_structure('experiment1')
    # experiment.preprocess_raw()
    # experiment.bin_binarise()
    # experiment.get_metadata()
    # experiment.evaluate_crowd_all_answers()
    # exit()
     # experiment.drop_analysis(N_Features, n_samples)
    #experiment.evaluate_flock(N_Features, n_samples, range(3, 100, 1))
    # experiment.evaluate_csfs_auc()

    # experiment.drop_analysis(N_Features, n_samples)
    # N_Features = [65, 80, 95, 116]
    # N_Features = [5, 17, 32, 50]
    # experiment.drop_evaluation(N_Features, n_samples)
    budget_range = range(10, 180, 10)
    no_features = range(1, 18)
    # experiment.evaluate_budget(budget_range)
    # df_budget_evaluation = pd.read_csv(experiment.path_budget_evaluation, index_col=0, header=[0, 1])
    # experiment.get_figure_budget_evaluation(df_budget_evaluation)
    # experiment.evaluate_ranking_cost(budget_range)
    # experiment.evaluate_ranking_nofeatures(N_Features)
    # experiment.autocorrelation()
    experiment.final_evaluation(feature_range=no_features, bootstrap_n=12, repetitions=10)
    experiment.final_evaluation_visualisation(feature_range=no_features)
    # experiment.crowd_answers_plot()
    # experiment.evaluate_csfs_auc()