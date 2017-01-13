import os
import numpy as np
import pandas as pd
import sys

import pickle
import scipy.stats as st
from joblib import Parallel, delayed
from tabulate import tabulate

import CSFSLoader
from CSFSCrowdCleaner import CSFSCrowdAggregator, CSFSCrowdCleaner, CSFSCrowdAnalyser, CSFSCrowdAnswergrouper
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestActualSelector, CSFSBestFromMetaSelector
from analysis_noisy_means_drop import _conduct_analysis, visualise_results
from application.CSFSConditionEvaluation import TestEvaluation
from application.EvaluationRanking import ERCondition, ERCostEvaluator, ERNofeaturesEvaluator
from infoformulas_listcomp import H, _H, IG_from_series
from util.util_features import get_features_from_questions
import pandas as pd
import numpy as np
import plotly.graph_objs as go

class AbstractExperiment:

    path_raw = ''
    path_cleaned = ''
    path_bin = ''
    path_meta = ''
    path_answers_raw = ''
    path_answers_clean = ''
    path_answers_clean_grouped = ''
    path_answers_aggregated = ''
    path_answers_metadata = ''
    path_csfs_auc = ''
    path_questions = ''
    path_flock_result = ''
    path_cost_ig_test = ''
    path_cost_ig_expert = ''
    path_cost_ig_base = ''
    path_budget_evaluation_cost = ''
    path_budget_evaluation_nofeatures = ''
    path_budget_evaluation_base = ''
    path_budget_evaluation_result = ''
    path_budget_evaluation_cost_rawaucs = ''
    path_budget_evaluation_nofeatures_rawaucs = ''
    path_final_evaluation_aucs = ''
    target = ''

    def __init__(self, dataset_name, experiment_number, experiment_name):
        self.dataset_name = dataset_name
        self.number = experiment_number
        self.experiment_name = experiment_name
        self.base_path = 'datasets/{}/'.format(self.dataset_name)

    def _create_if_nonexisting(self, path, folder):
            if folder not in os.listdir(path):
                os.mkdir('{}{}'.format(path, folder))

    def set_up_basic_folder_structure(self):
        default_folders = ['cleaned', 'raw', 'questions', 'results']

        for folder in default_folders:
            self._create_if_nonexisting(self.base_path, folder)

    def set_up_experiment_folder_structure(self, experiment_name):
        default_folders = ['cleaned', 'raw', 'questions', 'results']

        folder_name = experiment_name

        for folder in default_folders:
            path = '{}{}/'.format(self.base_path, folder)
            self._create_if_nonexisting(path, folder_name)

        self._create_if_nonexisting('{}raw/'.format(self.base_path), 'default')



    def explore_original(self):
        """
        Outputs a python notebook with H, Ig, Ig ratio in "raw" folder
        :return:
        """
        pass

    def preprocess_raw(self):
        """
        Selects only interesting features, fills gaps
        outputs a csv into "cleaned" folder "_clean"
        :return:
        """
        pass

    def bin_binarise(self):
        """
        binning and binarise
        outputs a csv into "cleaned" folder "_bin"
        :return:
        """
        pass

    def get_metadata(self):
        """
        Outputs a csv with p, p|f=0, p|f=1, H, Ig, Ig ratio in "cleaned" folder
        :return:
        """
        df_data = pd.read_csv(self.path_bin)
        # df_data = df_data[df_data['subject'] == 0] # limit data to certain subject.

        df = pd.DataFrame()
        df['mean'] = np.mean(df_data)

        def cond_mean(df, cond_value, target):
            result = list()
            for f in df:
                tmp_df = df[df[f] == cond_value]
                result.append(np.mean(tmp_df[target]))
            return result

        df['mean|f=0'] = cond_mean(df_data, cond_value=0, target=self.target)
        df['mean|f=1'] = cond_mean(df_data, cond_value=1, target=self.target)
        df['std'] = np.std(df_data)

        df['H'] = [H(df_data[x]) for x in df_data]
        h_x = _H([df.loc[self.target]['mean'], 1-df.loc[self.target]['mean']])
        df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='mean')
        df['IG ratio'] = df.apply(lambda x: x['IG']/x['H'], axis='columns') # correct?
        df = df.sort_values(by=['IG'], ascending=False)
        df.to_csv(self.path_meta, index=True)

    def _remove_non_informative_rows(self, df, threshold):
        """
        returns row indices where more than threshold entries are missing, e.g. 0.5
        :param threshold: ratio, e.g. 0.5
        """
        df_tmp = pd.DataFrame()
        n_features = len(df.columns)
        # calculating ratio of rows that have more than "ratio" missing values
        df_tmp['ratio'] = df.apply(lambda row: row.isnull().sum()/n_features, axis='columns')

        # kick too noisy rows
        return df[df_tmp['ratio'] <= threshold]

    def drop_analysis(self, N_features, n_samples=100):
        df = CSFSLoader.CSFSLoader().load_dataset(self.path_bin)
        Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, self.target, mean_error, N_features, n_samples, self.dataset_name) for mean_error in np.linspace(0.0, 0.6, 200))

    def drop_evaluation(self, N_features, n_samples):
        visualise_results(dataset_name=self.dataset_name, N_features=N_features, show_plot=False, N_samples=n_samples, dataset_class=self.experiment_name)


    def _get_dataset_bin(self):
        """
        Selects subset of data set we have questions for.
        """
        df_raw = pd.read_csv(self.path_bin)
        # kick all features we don't want
        features = get_features_from_questions(self.path_questions, remove_cond=True)
        features.append(self.target)
        df = df_raw[features]
        return df

    def _get_sample_df(self, df, features, r):
        """
        creates a dataframe with r samples for each feature
        :param df: df_crowd_answers. row must contain columns with feature name and column with answer
        :param features: list(str)
        :param r: int
        :return: df_sample
        """
        grouped = df.groupby('feature')
        df_sample = pd.DataFrame()
        for feature in features:
            group = grouped.get_group(feature)
            samples = group.sample(n=r)
            df_sample = df_sample.append(samples)
        return df_sample

    def evaluate_crowd_all_answers(self, mode=CSFSCrowdAggregator.Mode.EXTENDED, fake_features={}):
        """
        Aggregates crowd answers and evaluates for all crowd answers
        :return:
        """
        df_clean = CSFSCrowdCleaner(self.path_questions, self.path_answers_raw, self.target).clean()
        for f in fake_features:
            df_clean = df_clean.append({'answer': fake_features[f], 'answerUser': 'FAKE', 'feature': f}, ignore_index=True)
        df_clean.to_csv(self.path_answers_clean, index=True)

        df_clean_grouped = CSFSCrowdAnswergrouper.group(df_clean)
        df_clean_grouped.to_pickle(self.path_answers_clean_grouped)

        df_aggregated = CSFSCrowdAggregator(df_clean, target=self.target, mode=mode, fake_features=fake_features).aggregate()
        df_aggregated.to_csv(self.path_answers_aggregated, index=True)

        df_combined = CSFSCrowdAnalyser().get_combined_df(self.path_answers_aggregated, self.path_meta)
        df_combined.to_csv(self.path_answers_metadata, index=True)

    def _append_fake_user_answers(self, df, feature, value, n=1):
        """
        Appends n user answers for feature with given value to df
        :param feature: str
        :param value: float (e.g. median)
        :param n: int
        :return:df
        """
        data = {'answer': value, 'answerUser': 'FAKE', 'feature': feature}
        df = df.append([data]*n, ignore_index=True) # need to append it several times in order to allow random selection
        return df


    def evaluate_csfs_auc(self, fake_features={}, fake_till_n=-1):
        df_data = self._get_dataset_bin()
        evaluator = CSFSEvaluator(df_data, self.target)

        df_crowd_answers = pd.read_csv(self.path_answers_clean, index_col=0)
        min_count = df_crowd_answers.groupby('feature').agg('count').min().min() # returns number of responses for feature with fewest answers

        if fake_till_n > min_count:
            # make sure all features have at least fake_till_n entries by duplicating answers
            min_count = fake_till_n
            df_counts = df_crowd_answers.groupby('feature', as_index=False).agg('count')[['feature', 'answer']]
            df_counts['missing'] = df_counts['answer'].apply(lambda x: x-min_count)
            df_counts = df_counts[df_counts['missing']<0]
            for f in df_counts['feature']:
                median = np.median(df_crowd_answers[df_crowd_answers['feature']==f]['answer'])
                n = df_counts[df_counts['feature']==f]['missing'].values[0] * -1
                self._append_fake_user_answers(df_crowd_answers, f, median, n)
            feature_to_replicate = df_counts[df_counts['answer']<min_count]['feature']

        for f in fake_features:
            df_crowd_answers = self._append_fake_user_answers(df_crowd_answers, f, fake_features[f], min_count)

        R = range(3, min_count, 1) # number of samples
        N_Feat = [3, 5, 7, 9, 11]
        n_samples = 100 # number of repetitions to calculate mean auc (and std)

        df_csfs_auc = pd.DataFrame(index=R, columns=N_Feat)
        df_csfs_std = pd.DataFrame(index=R, columns=N_Feat)
        features = list(set(df_crowd_answers['feature']))
        for r in R:
            sys.stdout.write('r: {}\n'.format(r))
            aucs = {n_feat: list() for n_feat in N_Feat}

            for i in range(n_samples):
                # sample a number of crowd answers for each feature randomly
                df_sample = self._get_sample_df(df_crowd_answers, features, r)

                # get df with metadata that will make it possible to select n best features
                df_crowd_metadata = CSFSCrowdAggregator(df_sample, target=self.target).aggregate()
                # select features+
                selector = CSFSBestFromMetaSelector(df_crowd_metadata)

                for n_feat in N_Feat:
                    nbest_features = selector.select(n_feat)
                    auc = evaluator.evaluate_features(nbest_features)
                    aucs[n_feat].append(auc)

            df_csfs_auc.loc[r] = {n_feat: np.mean(aucs[n_feat]) for n_feat in N_Feat}
            df_csfs_std.loc[r] = {n_feat: np.std(aucs[n_feat]) for n_feat in N_Feat}
        # print(df_csfs_auc)
        df_csfs_auc.to_csv(self.path_csfs_auc)
        df_csfs_std.to_csv(self.path_csfs_std)



    def evaluate_flock(self, N_features, n_samples=100, R=range(3, 100, 1)):
        """

        :param N_features: list(int)
        :param n_samples: int
        :param R: list(int) Costs in HITs
        :return:
        """
        df_data = self._get_dataset_bin()
        evaluator = CSFSEvaluator(df_data, self.target)

        # R = range(3, 100, 1) # number of samples
        result = pd.DataFrame(columns=N_features, index=R)

        for r in R:
            sys.stdout.write('r: {}\n'.format(r))
            aucs = {n_feat: list() for n_feat in N_features}
            for i in range(n_samples):
                # get a number of samples
                df_sample = df_data.sample(n=r, axis='rows')
                df_sample.index = range(r) # otherwise we have a problem with automatic iteration when calculating conditional probabilities
                best_selector = CSFSBestActualSelector(df_sample, self.target)

                for n_feat in N_features:
                    nbest_features = best_selector.select(n_feat)
                    auc = evaluator.evaluate_features(nbest_features)
                    aucs[n_feat].append(auc)
            result.loc[r] = {n_feat: np.mean(aucs[n_feat]) for n_feat in aucs}
        result.to_csv(self.path_flock_result)

    # def evaluate_budget(self, budget_range):
    #     evaluation_test = TestEvaluation(self.path_cost_ig_test, self.path_bin, self.target)
    #     df_test = evaluation_test.get_auc_for_budget_range(budget_range)
    #
    #     evaluation_expert = RankingEvaluation(self.path_cost_ig_expert, self.path_bin, self.target)
    #     df_expert = evaluation_expert.get_auc_for_budget_range(budget_range)
    #
    #     df_result = pd.concat(dict(test=df_test, expert=df_expert), axis='columns')
    #     df_result.to_csv(self.path_budget_evaluation, index=True)

    def evaluate_ranking_cost(self, budget_range):
        """
        Creates a csv with auc, CI for each condition (1-4). index: cost
        :param budget_range:
        :return:
        """
        # load answer df and cost_ig df
        df_evaluation_result = pd.read_csv(self.path_budget_evaluation_result, header=None, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        df_evaluation_base = pd.read_csv(self.path_budget_evaluation_base)
        df_cleaned_bin = pd.read_csv(self.path_bin)

        evaluator = ERCostEvaluator(df_evaluation_result, df_evaluation_base, df_cleaned_bin, target=self.target, dataset_name=self.dataset_name)
        df_aucs_raw, data_evaluated = evaluator.evaluate_all(budget_range)

        df_test = TestEvaluation(self.path_cost_ig_test, self.path_bin, self.target).get_auc_for_budget_range(budget_range)

        data_evaluated[ERCondition.TEST] = df_test
        df_evaluated = pd.concat(data_evaluated, axis='columns')
        df_evaluated.to_csv(self.path_budget_evaluation_cost)

        df_aucs_raw.to_pickle(self.path_budget_evaluation_cost_rawaucs)

    def evaluate_ranking_nofeatures(self, feature_range):
        """
        Creates a csv with auc, CI for each condition (1-4). index: number of features (nofeatures)
        :return:
        """   # load answer df and cost_ig df
        df_evaluation_result = pd.read_csv(self.path_budget_evaluation_result, header=None, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        df_evaluation_base = pd.read_csv(self.path_budget_evaluation_base)
        df_cleaned_bin = pd.read_csv(self.path_bin)

        evaluator = ERNofeaturesEvaluator(df_evaluation_result, df_evaluation_base, df_cleaned_bin, target=self.target, dataset_name=self.dataset_name)
        df_aucs_raw, data_evaluated = evaluator.evaluate_all(feature_range)

        df_test = TestEvaluation(self.path_cost_ig_test, self.path_bin, self.target).get_auc_for_nofeatures_range(feature_range)

        data_evaluated[ERCondition.TEST] = df_test
        df_evaluated = pd.concat(data_evaluated, axis='columns')
        df_evaluated.to_csv(self.path_budget_evaluation_nofeatures)

        df_aucs_raw.to_pickle(self.path_budget_evaluation_nofeatures_rawaucs)

    def final_evaluation(self, feature_range):
        """
        Final evaluation. Takes tokens for condition 1-4 and outputs aucs for #features
        :param feature_range: list(int)
        :return: saves final_evaluation_aucs.pickle
        """
        df_evaluation_result = pd.read_csv(self.path_budget_evaluation_result, header=None, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        df_evaluation_base = pd.read_csv(self.path_budget_evaluation_base)
        df_cleaned_bin = pd.read_csv(self.path_bin)
        df_answers_grouped = pd.read_pickle(self.path_answers_clean_grouped)

        evaluator = ERNofeaturesEvaluator(df_evaluation_result, df_evaluation_base, df_cleaned_bin, target=self.target, dataset_name=self.dataset_name, df_answers_grouped=df_answers_grouped)
        raw_data, evaluated = evaluator.evaluate_all_to_dict(feature_range) # raw_data is dict: {CONDITION: {NOFEATURES: [AUCS]}}
        print(raw_data)
        pickle.dump(raw_data, open(self.path_final_evaluation_aucs, 'wb'))



    def get_figure_budget_evaluation(self, df_budget_evaluation):
        """
        plot (in python notebook):
        # plotly.offline.iplot(fig)
        :param df_cost_ig pd.DataFrame
        :return figure
        """
        def get_traces(df, name):
            trace_auc = go.Scatter(
                x=df.index,
                y=df.auc,
                name='AUC {}'.format(name)
                )
            # trace_fc = go.Bar(
            #     x=df.index,
            #     y=df.count_features_ratio,
            #     name='ratio #features selected {}'.format(name)
            # )
            # return [trace_auc, trace_fc]
            return trace_auc

        data = list()
        for header in set(df_budget_evaluation.columns.get_level_values(0)):  # returns only columns on level 0 (test, expert,...)
            traces = get_traces(df_budget_evaluation[header], header)
            data.append(traces[0])
            #data.append(traces[1])


        layout = go.Layout(
            title='Performance vs. Budget',
            xaxis=dict(
                title='Cost',

            ),
            yaxis=dict(
                range=[0, 1],
            ),
        )

        fig = go.Figure(data=data, layout=layout)
        return fig

