import os
import pickle
import sys

import math
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.weightstats as ssw
from joblib import Parallel, delayed
import plotly
import plotly.graph_objs as go
from scipy import stats
from tabulate import tabulate

import CSFSLoader
from CSFSCrowdCleaner import CSFSCrowdAggregator, CSFSCrowdCleaner, CSFSCrowdAnalyser, CSFSCrowdAnswergrouper
from CSFSEvaluator import CSFSEvaluator
from CSFSSelector import CSFSBestActualSelector, CSFSBestFromMetaSelector
from FinalEvaluation import FinalEvaluationCombiner
from analysis_noisy_means_drop import _conduct_analysis, visualise_results
from application.CSFSConditionEvaluation import TestEvaluation
from application.EvaluationRanking import ERCondition, ERCostEvaluator, ERNofeaturesEvaluator
from csfs_stats import hedges_g
from csfs_visualisations import CIVisualiser, AnswerDeltaVisualiserLinePlot, \
    AnswerDeltaVisualiserBar, AnswerDeltaVisualiserBox
from humans_vs_actual_auc import FeatureRankerAUC, FeatureCombinationCalculator
from infoformulas_listcomp import H, _H, IG_from_series
from util.util_features import get_features_from_questions


class AbstractExperiment:

    path_raw = ''
    path_cleaned = ''
    path_bin = ''
    path_autocorrelation = ''
    path_meta = ''
    path_answers_raw = ''
    path_answers_clean = ''
    path_answers_clean_grouped = ''
    path_answers_plots = ''
    path_answers_aggregated = ''
    path_answers_metadata = ''
    path_answers_delta = ''
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
    path_final_evaluation_aggregated = ''
    path_final_evaluation_combined = ''
    path_auc_plots = ''
    path_comparison = ''
    path_no_answers_vs_auc = ''
    path_humans_vs_actual_auc = ''
    target = ''
    answer_range = range(1, 17)
    feature_range = range(1, 2)
    bootstrap_n = 9
    repetitions = 19

    sec = 80
    x = 1
    # condition -> colour
    colours = {1:'rgba(255, {}, {}, {})'.format(sec, sec, x),
            2: 'rgba( {}, 255,  {}, {})'.format(sec, sec, x),
                3: 'rgba( {},  {}, 255, {})'.format(sec, sec, x),
                    4: 'rgba(0, 0, 0, {})'.format(x),
                        5: 'rgba(100, 100, 100, {})'.format(x),
                            6: 'rgba(200, 150, 0, {})'.format(x),
                        }

    def __init__(self, dataset_name, experiment_number, experiment_name):
        self.dataset_name = dataset_name
        self.number = experiment_number
        self.experiment_name = experiment_name
        self.base_path = 'datasets/{}/'.format(self.dataset_name)
        self.path_comparison = '{}evaluation/comparison/'.format(self.base_path)
        self.path_answers_delta_plot_box = '{}results/{}/visualisations/{}_answers_delta_plot_box.html'.format(self.base_path, experiment_name, self.dataset_name)
        self.path_answers_delta_plot_line = '{}results/{}/visualisations/{}_answers_delta_plot_line.html'.format(self.base_path, experiment_name, self.dataset_name)
        self.path_humans_vs_actual_auc = '{}evaluation/comparison/humans_vs_actual_auc.json'.format(self.base_path)

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

    def crowd_answers_plot(self, auto_open=False):
        """
        Plots three bar charts for each feature showing the distribution of the answers for p, p|f=0 and p|f=1
        :return:
        """
        def get_trace(series, condition):
            return go.Histogram(
                x=list(series),
            )

        df_answers_grouped = pd.read_pickle(self.path_answers_clean_grouped)

        features = list(df_answers_grouped.index)
        conditions = ['p', 'p|f=0', 'p|f=1']
        subplot_titles = ["{} {}".format(f, c) for f in features for c in conditions ]
        fig = plotly.tools.make_subplots(rows=len(features), cols=3, subplot_titles=subplot_titles)
        row_index=1
        for index, row in df_answers_grouped.iterrows():
            for i in range(len(conditions)):
                trace = get_trace(row[conditions[i]], conditions[i])
                fig.append_trace(trace, row_index, i+1)
            row_index += 1
        fig['layout'].update(showlegend=False, height=2500, width=1200, title='Crowd Answers for {} ({})'.format(self.dataset_name, self.experiment_name),)

        plotly.offline.plot(fig, auto_open=auto_open, filename=self.path_answers_plots)
        from IPython.display import Image
        Image('a-simple-plot.png')

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

    def autocorrelation(self):
        """
        Calculcates Kendall-Tau Correlation for binary features
        :return:
        """
        # For all features calculate kendall's tau with every other feature.
        df_bin = pd.read_csv(self.path_bin)
        features = sorted(list(df_bin.columns))
        df_correlation = pd.DataFrame({f: [np.nan] * len(features) for f in features}, index=features)
        for f1 in features:
            for f2 in features:
                x = list(df_bin[f1])
                y = list(df_bin[f2])
                corr, p = scipy.stats.kendalltau(x, y)
                df_correlation.loc[f1, f2] = "{} (p={:.3f})".format(corr, p)
                if f1 == f2:
                    break
        df_correlation.to_csv(self.path_autocorrelation, index=True)

    def final_evaluation(self):
        """
        Comparing conditions for a given number of answers for csfs
        Final evaluation. Takes tokens for condition 1-6 and outputs aucs for #features
        :param feature_range: list(int)
        :return: saves final_evaluation_aucs.pickle
        """
        df_evaluation_result = pd.read_csv(self.path_budget_evaluation_result, header=None, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        df_evaluation_base = pd.read_csv(self.path_budget_evaluation_base)
        df_cleaned_bin = pd.read_csv(self.path_bin)
        df_answers_grouped = pd.read_pickle(self.path_answers_clean_grouped)
        df_actual_metadata = pd.read_csv(self.path_answers_metadata, index_col=0, header=[0, 1])
        df_actual_metadata = df_actual_metadata['actual']
        evaluator = ERNofeaturesEvaluator(df_evaluation_result, df_evaluation_base, df_cleaned_bin, df_actual_metadata=df_actual_metadata, target=self.target, dataset_name=self.dataset_name, df_answers_grouped=df_answers_grouped, bootstrap_n=self.bootstrap_n, repetitions=self.repetitions, replace=False)
        raw_data = evaluator.evaluate_all_to_dict(self.feature_range) # raw_data is dict: {CONDITION: {NOFEATURES: [AUCS]}}
        pickle.dump(raw_data, open(self.path_final_evaluation_aucs, 'wb'))

    def crowd_auc_plot(self, auto_open=False):
        """
        Plots a bar chart for each number of features and condition showing the distribution of AUCs
        :return:
        """
        def get_name(nofeat, cond):
            plural = nofeat > 1
            return "{} features (condition {})".format(nofeat, cond) if plural else "{} feature (condition {})".format(nofeat, cond)

        def get_trace(values, nofeat, cond):
            name = get_name(nofeat, cond)
            return go.Histogram(
                name=name,
                x=values,
                histnorm='probability',
                autobinx=False,
                xbins=dict(
                    start=0.5,
                    end=1,
                    size=0.025,
                    ),
                marker=dict(
                    color=self.colours[cond]
                ),)

        aucs = pd.read_pickle(self.path_final_evaluation_aucs)
        nofeatures = sorted(set([nofeat for nofeat in aucs[1]]))
        conditions = [1, 2, 3, 4, 5]
        subplot_titles = [get_name(no_feat, c) for no_feat in nofeatures for c in conditions]
        fig = plotly.tools.make_subplots(rows=len(nofeatures), cols=len(conditions), subplot_titles=subplot_titles)
        row_index=1
        for no_feat in nofeatures:
            for i in range(len(conditions)):
                trace = get_trace(aucs[conditions[i]][no_feat], no_feat, conditions[i])
                fig.append_trace(trace, row_index, i+1)
            row_index += 1
        fig['layout'].update(showlegend=False, height=2500, width=1200, title='AUC Histograms for {} ({})'.format(self.dataset_name, self.experiment_name),)

        plotly.offline.plot(fig, auto_open=auto_open, filename=self.path_auc_plots)
        from IPython.display import Image
        Image('a-simple-plot.png')

    def final_evaluation_visualisation(self, feature_range):
        raw_data = pickle.load(open(self.path_final_evaluation_aucs, 'rb'))
        data_aggregated = dict()
        for condition in raw_data:
            data = {
                'mean': [np.mean(raw_data[condition][nofeature]) for nofeature in raw_data[condition]],
                'ci_lo': [ssw.DescrStatsW(raw_data[condition][nofeature]).tconfint_mean()[0] for nofeature in raw_data[condition]],
                'ci_hi': [ssw.DescrStatsW(raw_data[condition][nofeature]).tconfint_mean()[1] for nofeature in raw_data[condition]],
                'std': [np.std(raw_data[condition][nofeature]) for nofeature in raw_data[condition]],
                'count': [np.count_nonzero(raw_data[condition][nofeature]) for nofeature in raw_data[condition]],
            }

            df = pd.DataFrame(data)
            data_aggregated[condition] = df
        df_combined = pd.concat(data_aggregated, axis='columns')
        df_combined.index = feature_range
        df_combined.to_pickle(self.path_final_evaluation_aggregated)

    def final_evaluation_combine(self, feature_range, bootstrap_n=12, repetitions=20):
        """
        Combines conditions 1-4 to a file according to patrick's wishes:
        number_of_features, dataset, ranking_strategy, user_id, AUC, AUC_95_CI_low, AUC_95_CI_high
        :param feature_range: list(int)
        :return:
        """
        df_evaluation_result = pd.read_csv(self.path_budget_evaluation_result, header=None, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        df_evaluation_base = pd.read_csv(self.path_budget_evaluation_base)
        df_cleaned_bin = pd.read_csv(self.path_bin)
        df_answers_grouped = pd.read_pickle(self.path_answers_clean_grouped)

        combiner = FinalEvaluationCombiner(df_evaluation_result, df_evaluation_base, df_cleaned_bin, target=self.target, dataset_name=self.dataset_name, df_answers_grouped=df_answers_grouped, bootstrap_n=bootstrap_n, repetitions=repetitions)
        df_combined = combiner.combine(feature_range)
        df_combined.to_csv(self.path_final_evaluation_combined, index=False)

    def statistical_comparison(self, feature_range):
        def get_p_str(p):
            if p >= 0.05:
                p_str = "p"
            elif 0.01 <= p < 0.05:
                p_str = "p*"
            else:
                p_str = "p**"
            return "{}={:.4f}".format(p_str, p)


        def get_df_compared(aucs, target_condition):
            df = pd.DataFrame(columns=feature_range, index=conditions)
            for c in conditions:
                for no_feat in feature_range:
                    a = aucs[c][no_feat]
                    b = aucs[target_condition][no_feat]
                    t, p = scipy.stats.ttest_ind(a, b, equal_var=False)
                    g = hedges_g(a, b)
                    t_str = "t={:.4f}".format(t)
                    p_str = get_p_str(p)
                    g_str = "g={:.4f}".format(g)
                    value = "{} {} {}".format(t_str, p_str, g_str)
                    df.loc[c, no_feat] = value

            df.columns = ["f_count={}".format(no_feat) for no_feat in feature_range]
            df.index = ["condition={}".format(ERCondition.get_string_identifier(c)) for c in conditions]
            return df

        conditions = [1, 2, 3, 4, 5]
        aucs = pd.read_pickle(self.path_final_evaluation_aucs)
        for c in conditions:
            df = get_df_compared(aucs, c)
            print(tabulate(df, headers='keys'))
            path_out = "{}{}_{}-vs-others.csv".format(self.path_comparison, self.dataset_name, ERCondition.get_string_identifier(c))
            df.to_csv(path_out, index=True)

    def evaluate_no_answers(self):
        """
        Creates plots for answers in answer_range. samples answers without replacement and calculcates AUC
        { 2 features: {2answers: [], 3 answers: [], 4 answers: [],...}, 3 features: [2answers:[], 3answers:[]},...}
        only for CSFS
        :param feature_range:
        :param answer_range:
        :return:
        """
        answer_range = self.answer_range
        feature_range = self.feature_range
        repetitions = self.repetitions

        df_cleaned_bin = pd.read_csv(self.path_bin)
        df_answers_grouped = pd.read_pickle(self.path_answers_clean_grouped)
        df_actual_metadata = pd.read_csv(self.path_answers_metadata, index_col=0, header=[0, 1])
        df_actual_metadata = df_actual_metadata['actual']

        # # feature_range = [2,3]
        # # answer_range = [2,10]
        # repetitions=5

        result = {}
        for no_answers in answer_range:
            print('calculating. number of answers: ', no_answers)
            evaluator = ERNofeaturesEvaluator(None, None, df_cleaned_bin, df_actual_metadata=df_actual_metadata, target=self.target, dataset_name=self.dataset_name, df_answers_grouped=df_answers_grouped, bootstrap_n=no_answers, repetitions=repetitions, replace=False)
            raw_data = evaluator.evaluate(feature_range, condition=ERCondition.CSFS) # raw_data is dict: {CONDITION: {NOFEATURES: [AUCS]}}
            result[no_answers] = raw_data[ERCondition.CSFS]

        # result is dict: {no_answers: {NOFEATURES: [AUCS]}}
        result_restructured = dict()
        for no_features in feature_range:
            result_restructured[no_features] = {no_answers: result[no_answers][no_features] for no_answers in answer_range}
        # {no_features: {no_answers: result[no_answers][no_features]} for no_features in feature_range for no_answers in answer_range }
        result = result_restructured # { 2 features: {2answers: [], 3 answers: [], 4 answers: [],...}, 3 features: [2answers:[], 3answers:[]},...}

        # print(result)
        data_aggregated = dict()
        for no_features in result:
            print('aggregating. number of features: ', no_features)
            data = {
                'mean': [np.mean(result[no_features][no_answers]) for no_answers in answer_range],
                'ci_lo': [ssw.DescrStatsW(result[no_features][no_answers]).tconfint_mean()[0] for no_answers in answer_range],
                'ci_hi': [ssw.DescrStatsW(result[no_features][no_answers]).tconfint_mean()[1]for no_answers in answer_range],
                'std': [np.std(result[no_features][no_answers]) for no_answers in answer_range],
            }

            df = pd.DataFrame(data)
            # print(no_features)
            # print(tabulate(df))
            data_aggregated[no_features] = df
        df_combined = pd.concat(data_aggregated, axis='columns')
        # exit()
        df_combined.index = answer_range
        df_combined.to_pickle(self.path_no_answers_vs_auc)

    def evaluate_no_answers_get_fig(self, feature_range, path_prefix=""):
        """
        Returns figure for a certain feature range
        :param feature_range: list(int)
        :return:
        """
        filename = '{}final_evaluation/{}_answers-vs-auc.html'.format(path_prefix, self.dataset_name)
        df = pd.read_pickle(path_prefix+self.path_no_answers_vs_auc)
        fig = CIVisualiser.get_fig(df, feature_range, 'number of answers sampled (without replacement)', y_title='AUC', title="{} ({} repetitions)".format(self.dataset_name, self.repetitions))
        # plotly.offline.plot(fig, auto_open=True, filename=filename)
        return fig

    def evaluate_answers_delta(self):
        def calc_ig(row, p_target):
            h_x = _H([p_target, 1-p_target])
            row['IG'] = IG_from_series(row, h_x=h_x, identifier='p')
            return row

        def get_avg_values(no_answers, df_answers_grouped, df_actual, p_target):
            conditions = ['p', 'p|f=0',  'p|f=1', 'IG']
            delta = {c: list() for c in conditions}

            for i in range(self.repetitions):
                df_sampled = pd.DataFrame(index=df_answers_grouped.index, columns=df_answers_grouped.columns)
                for condition in df_answers_grouped.columns: # p, p|f=0 and p|f=1
                    df_sampled[condition] = df_answers_grouped[condition].apply(lambda l: np.random.choice(list(l), no_answers, replace=False))
                    df_sampled[condition] = df_sampled[condition].apply(lambda l: np.median(l))
                df_sampled = df_sampled.apply(calc_ig, axis='columns', p_target=p_target)
                """
                                 p  p|f=0  p|f=1        IG
        Fjob_teacher           0.1    0.5    0.7  0.011871
        Medu_(-0.004, 1.333]   0.2    0.6    0.5  0.023240
        Mjob_at_home           0.3    0.5    0.5  0.000000
        Mjob_other             0.2    0.7    0.5  0.094967
                """
                df_actual.columns = df_sampled.columns # rename from 'mean', 'mean|f=0', 'mean|f=1', 'IG' to 'p', 'p|f=0' and 'p|f=1', 'IG'
                df_diff = abs(df_actual - df_sampled)
                # pd.concat({'actual': df_actual, 'sampled': df_sampled, 'diff': df_diff}, axis='columns').to_csv('df_compare.csv', index=True)
                # exit()

                for c in conditions:
                    delta[c].append(np.mean(df_diff[c]))
            series = pd.Series(delta)
            return series

        df_answers_grouped = pd.read_pickle(self.path_answers_clean_grouped)
        p_target = df_answers_grouped['p'].loc[self.target][0]
        df_answers_grouped = df_answers_grouped.drop(self.target)
        df_answers_grouped = df_answers_grouped.sort_index()
        df_actual = pd.read_csv(self.path_answers_metadata, index_col=0, header=[0, 1])['actual'].drop(self.target)
        df_actual = df_actual.sort_index()

        df_result = pd.DataFrame({no_answers: get_avg_values(no_answers, df_answers_grouped, df_actual, p_target) for no_answers in self.answer_range}).transpose()
        df_result.to_pickle(self.path_answers_delta)

    def evaluate_answers_delta_plot(self, auto_open=False):
        auto_open=True
        df = pd.read_pickle(self.path_answers_delta)
        title = '{}: Number of Answers versus Error ({} Repetitions)'.format(self.dataset_name, self.repetitions)
        # fig = AnswerDeltaVisualiserLinePlot(title=title).get_figure(df)
        # plotly.offline.plot(fig, auto_open=auto_open, filename=self.path_answers_delta_plot_line)
        # import time
        # time.sleep(2) # delays for 5 seconds
        fig = AnswerDeltaVisualiserBox(title=title).get_figure(df)
        plotly.offline.plot(fig, auto_open=auto_open, filename=self.path_answers_delta_plot_box)

    def humans_vs_actual_auc(self, mode='combination', merge_combinations=True):
        df_evaluation_result = pd.read_csv(self.path_budget_evaluation_result, header=None, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        df_evaluation_base = pd.read_csv(self.path_budget_evaluation_base)
        df_cleaned_bin = pd.read_csv(self.path_bin)
        df_answers_grouped = pd.read_pickle(self.path_answers_clean_grouped)
        evaluator = ERNofeaturesEvaluator(df_evaluation_result, df_evaluation_base, df_cleaned_bin, df_actual_metadata=None, target=self.target, dataset_name=self.dataset_name, df_answers_grouped=df_answers_grouped, bootstrap_n=self.bootstrap_n, repetitions=self.repetitions)
        features = list(pd.read_csv(self.path_answers_metadata, index_col=0, header=[0, 1]).index)
        features.remove(self.target)
        values_domain = evaluator.evaluate(self.feature_range, ERCondition.DOMAIN)[ERCondition.DOMAIN]
        values_experts = evaluator.evaluate(self.feature_range, ERCondition.EXPERT)[ERCondition.EXPERT]
        values_lay = evaluator.evaluate(self.feature_range, ERCondition.LAYPERSON)[ERCondition.LAYPERSON]

        if mode == 'rank':
            ranker = FeatureRankerAUC(df_cleaned_bin, self.target, features)
            values_worst = ranker.get_ranked(reverse=True, return_features=False)
            values_best = ranker.get_ranked(reverse=False, return_features=False)

        else:
            if merge_combinations:
                # combinations have already been calculated. get them from file to save time
                data = pd.read_json(self.path_humans_vs_actual_auc)
                values_best = data['best']
                values_worst = data['worst']
            else:
                calculator = FeatureCombinationCalculator(df_cleaned_bin, self.target, features)
                print('start best')
                values_best = calculator.get_aucs_for_feature_range(self.feature_range, reverse=False)
                print('start worst')
                values_worst = calculator.get_aucs_for_feature_range(self.feature_range, reverse=True)

        df_result = pd.DataFrame({'lay': values_lay, 'domain': values_domain, 'experts': values_experts, 'best': values_best, 'worst': values_worst })
        df_result.to_json(self.path_humans_vs_actual_auc)










