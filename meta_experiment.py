import json
import math

import itertools
import pandas as pd
import numpy as np
import pickle

import plotly
from functools import reduce
from tabulate import tabulate

from application.CSFSConditionEvaluation import AUCForOrderedFeaturesCalculator
from application.EvaluationRanking import ERCondition, ERFilterer, ERParser
from csfs_visualisations import HumanVsActualBarChart, AnswerDeltaVisualiserBox, HumanComparisonBarChart, \
    CSFSVsHumansBarChart, ClassifiersComparisonBarChart, CSFSVsHumansBarChart3
from experiment_income import ExperimentIncome
from experiment_olympia import ExperimentOlympia
from experiment_student_por import ExperimentStudent
from table_effect_size import EffectSizeTable, EffectSizeSingle, EffectSizeMatrix
from scipy.stats import spearmanr

from util.util_features import remove_binning_cond_markup, get_features_from_questions


class MetaExperiment:
    def __init__(self):
        self.path_final_evaluation_combined = 'final_evaluation/combined.csv'
        self.path_upwork_participants = 'participants.csv'
        self.path_statistical_comparison = 'final_evaluation/comparisons/' # e.g. + 'student_1-vs-2'
        self.path_human_vs_actual_histogram = 'paper_plots-and-data/human-vs-actual/human_vs_actual_hist.html'
        self.path_human_vs_actual_barchart = 'paper_plots-and-data/human-vs-actual/human_vs_actual_barchart.html'
        self.path_human_vs_actual_data = 'paper_plots-and-data/human-vs-actual/human_vs_actual_data.json'

        self.path_human_comparison_data = 'paper_plots-and-data/human-comparison/'
        self.path_human_comparison_plot = 'paper_plots-and-data/human-comparison/human-comparison.html'

        self.path_plot_no_answers_vs_delta_png = 'no_answers_vs_delta'
        self.path_plot_no_answers_vs_delta_html = 'paper_plots-and-data/answers-delta/no_answers_vs_delta_data.html'
        self.path_plot_no_answers_vs_delta_data = 'paper_plots-and-data/answers-delta/no_answers_vs_delta_data.json'

        self.path_auc_all_conditions = 'paper_plots-and-data/evaluation_all_conditions/'

        self.path_csfs_vs_humans_plot_nb = 'paper_plots-and-data/krowdd_vs_humans/krowdd_vs_humans_nb.html'
        self.path_csfs_vs_humans_plot_dt = 'paper_plots-and-data/krowdd_vs_humans/krowdd_vs_humans_dt.html'
        self.path_csfs_vs_humans_plot_mlp = 'paper_plots-and-data/krowdd_vs_humans/krowdd_vs_humans_mlp.html'
        self.path_csfs_vs_humans_plot_with_classifiers = 'paper_plots-and-data/krowdd_vs_humans/krowdd_vs_humans_with_classifiers.html'

        self.path_csfs_vs_humans_data = 'paper_plots-and-data/krowdd_vs_humans/'

        self.path_single_human_performance_data = 'final_evaluation/private_single_human_performance.json'
        self.path_data_scientists = 'final_evaluation/private_participants.csv'
        self.path_data_scientists_performance = 'final_evaluation/private_data-scientists_performance.json'

        self.path_chosen_features_ig = 'paper_plots-and-data/chosen_features_ig/'

        self.path_compare_classifiers = 'final_evaluation/compare_classifiers/'

        self.path_data = 'paper_plots-and-data/datasets/'

        self.ds_student = ExperimentStudent('student', 2, 'experiment2_por')
        self.ds_income = ExperimentIncome('income', 1, 'experiment1')
        self.ds_olympia = ExperimentOlympia('olympia', 4, 'experiment2-4_all')

        self.datasets = {'Student': self.ds_student, 'Income': self.ds_income, 'Olympics': self.ds_olympia}

    def get_evaluation_data(self, classifier='nb'):
        """
        Returns dict with key: dataset name as in Paper and value: pd.DataFrame with columns=Conditions (names from paper, e.g. Data Scientists) and rows=NoFeatures
        :return: dict
        """
        return {ds: pd.read_json("{}{}_evaluated_{}.json".format(self.path_data, ds, classifier)) for ds in self.datasets}

    def final_evaluation_combine_all(self):
        # for patrick
        df_student = pd.read_csv(self.ds_student.path_final_evaluation_combined)
        df_income = pd.read_csv(self.ds_income.path_final_evaluation_combined)
        df_olympia = pd.read_csv(self.ds_olympia.path_final_evaluation_combined)

        df_combined_all = pd.concat([df_student, df_income, df_olympia])

        # df_participants = pd.read_csv()
        df_combined_all.to_csv(self.path_final_evaluation_combined, index=False)

    def plot_humans_vs_actual_all_plot(self):
        """
        Plots human experts (domain and data science) relative to max/min performance for all datasets.
        :return:
        """
        df_student = pd.read_json(self.ds_student.path_humans_vs_actual_auc_nb).sort_index()
        df_income = pd.read_json(self.ds_income.path_humans_vs_actual_auc_nb).sort_index()
        df_olympia = pd.read_json(self.ds_olympia.path_humans_vs_actual_auc_nb).sort_index()
        df_all = [df_student, df_income, df_olympia]
        max_feature_count = max(list(df_student.index) + list(df_olympia.index) + list(df_income.index))
        range_features = range(1, max_feature_count+1)
        conditions = ['domain', 'experts', 'random'] # 'lay',

        def normalise(x, min, max):
            if x == min == max:
                z = -1
            # only for debugging
            elif min == max:
                return -1
            else:
                z = (x - min) / (max - min)
            # print(x,min,max,z)
            assert 0 <= round(z,3) <= 1 or z == -1

            return z

        def normalise_row(row):
            """
            Returns row with normalised values. only returns human conditions (others are 1 and 0, respectively)
            """
            row_norm = row[conditions].copy()
            for c in conditions:
                values_normalised = [normalise(x, min=row['worst'], max=row['best']) for x in row[c]]
                # print('vals')
                # print(values_normalised)
                values_normalised = [v for v in values_normalised if v != -1]
                # print(values_normalised)
                row_norm[c] = values_normalised
            return row_norm

        df_result = pd.DataFrame({c: {i: list() for i in range_features} for c in conditions}) # initialize dataframe with lists
        df_norm_all = [df.apply(normalise_row, axis='columns') for df in df_all]
        for df in df_norm_all:
            for column in df:
                for index in df[column].index: # list scores is array with float values
                    if index in df.index:
                        df_result.loc[index, column] += df.loc[index, column]
                    # print(df.loc[index, column])

        # print(tabulate(df_result, headers='keys'))
        #
        # exit()
        # df_result has index=range_answers and columns 'domain', 'experts',... values are lists of normalised scores
        def filter_row(row):
            # filters na and -1 values
            def filter(l):
                if isinstance(l, list):
                    return [e for e in l if 0 <= round(e,3) <= 1]
                if np.isnan(l):
                    return list()
            row = row.apply(filter)
            return row
            # filter na and -1 values
        df_result = df_result.apply(filter_row)
        df_result.columns = ['Domain Experts', 'Data Scientists', 'Random'] #'Laypeople'
        df_result.to_json(self.path_human_vs_actual_data)

        fig = HumanVsActualBarChart().get_figure(df_result, feature_range=range(1,10))
        plotly.offline.plot(fig, auto_open=True, filename=self.path_human_vs_actual_barchart)
        # fig = HumanVsActualBarChart().get_histograms(df_result)
        # plotly.offline.plot(fig, auto_open=True, image='png', filename=self.path_human_vs_actual_histogram)

    def plot_humans_vs_actual_all_plot2(self):
        """
        Plots human experts (domain and data science) relative to max/min performance for all datasets.
        :return:
        """
        data = self.get_evaluation_data()
        range_features = range(1, 10)
        conditions = ['Domain Experts', 'Data Scientists', 'Random'] # 'lay',

        def normalise(x, min, max):
            if x == min == max:
                z = -1
            # only for debugging
            elif min == max:
                return -1
            else:
                z = (x - min) / (max - min)
            # print(x,min,max,z)
            assert 0 <= round(z,1) <= 1 or z == -1

            return z

        def normalise_row(row):
            """
            Returns row with normalised values. only returns human conditions (others are 1 and 0, respectively)
            """
            row_norm = row[conditions].copy()
            print(row)
            for c in conditions:

                values_normalised = [normalise(x, min=row['Worst'], max=row['Best']) for x in row[c]]
                # print('vals')
                # print(values_normalised)
                values_normalised = [v for v in values_normalised if v != -1]
                # print(values_normalised)
                row_norm[c] = values_normalised
            return row_norm

        df_result = pd.DataFrame({c: {i: list() for i in range_features} for c in conditions}) # initialize dataframe with lists
        data_norm_all = {ds: data[ds].loc[range_features].apply(normalise_row, axis='columns') for ds in data}
        for ds in data_norm_all:
            df = data_norm_all[ds]
            for column in df:
                for index in df[column].index: # list scores is array with float values
                    if index in df.index:
                        df_result.loc[index, column] += df.loc[index, column]
                    # print(df.loc[index, column])

        # print(tabulate(df_result, headers='keys'))
        #
        # exit()
        # df_result has index=range_answers and columns 'domain', 'experts',... values are lists of normalised scores
        def filter_row(row):
            # filters na and -1 values
            def filter(l):
                if isinstance(l, list):
                    return [e for e in l if 0 <= round(e,3) <= 1]
                if np.isnan(l):
                    return list()
            row = row.apply(filter)
            return row
            # filter na and -1 values
        df_result = df_result.apply(filter_row)
        # df_result.columns = ['Domain Experts', 'Data Scientists', 'Random'] #'Laypeople'
        # df_result.to_json(self.path_human_vs_actual_data)

        fig = HumanVsActualBarChart().get_figure(df_result, feature_range=range(1,10))
        plotly.offline.plot(fig, auto_open=True, filename=self.path_human_vs_actual_barchart)

    def plot_no_answers_vs_delta(self):
        """
        Combined Boxplot for number of answers versus combined error for all three conditions.
        :return:
        """
        range_answers = range(1, 17)
        conditions = ['p', 'p|f=0', 'p|f=1']
        df_student = pd.read_pickle(self.ds_student.path_answers_delta)
        df_income = pd.read_pickle(self.ds_income.path_answers_delta)
        df_olympia = pd.read_pickle(self.ds_olympia.path_answers_delta)

        df_all = [df_student, df_income, df_olympia]
        # print(df_all[0])
        # exit()

        data = dict()
        for no_answers in range_answers:
            data[no_answers] = list()
            for df in df_all:
                for cond in conditions:
                    data[no_answers] += df.loc[no_answers, cond]

        df_no_answers_vs_delta = pd.DataFrame(data) # columns are no_answers
        # print(df_no_answers_vs_delta)
        # exit()
        df_no_answers_vs_delta.to_json(self.path_plot_no_answers_vs_delta_data)
        # 'Number of Answers versus Actual Data (19 Repetitions)'
        fig = AnswerDeltaVisualiserBox(title="").get_figure(df_no_answers_vs_delta)
        plotly.offline.plot(fig, auto_open=True, filename=self.path_plot_no_answers_vs_delta_html, image='png', image_filename=self.path_plot_no_answers_vs_delta_png)

    def table_kahneman(self):
        """
        Create table to check whether answers for conditional means were worse than others. (GitHub issue #30)
        :return:
        """
        range_answers = range(1, 17)
        conditions = ['p', 'p|f=0', 'p|f=1']
        datasets = {
            'Portuguese': pd.read_pickle(self.ds_student.path_answers_delta),
            'Income': pd.read_pickle(self.ds_income.path_answers_delta),
            'Olympics': pd.read_pickle(self.ds_olympia.path_answers_delta)
        }
        df_data = pd.DataFrame() # df with columns: Datset name and index: P(X=1),... values are lists
        for dataset in sorted(datasets):
            df = datasets[dataset]
            df_data[dataset] = pd.Series({
                'P(X=1)': df.loc[9, 'p'],
                'P(X=1|Y=0)': df.loc[9, 'p|f=0'],
                'P(X=1|Y=1)': df.loc[9, 'p|f=1']
            })
        def get_value(list):
            s = "{:.3f} (std={:.3f})".format(np.mean(list), np.std(list))
            return s
        df_result = df_data.apply(lambda r: r.apply(get_value))
        print(df_result)

        print('Delta Income vs. Portuguese + Olympics')
        for cond in df_data.index:
            a = df_data.loc[cond, 'Income']
            b = df_data.loc[cond, 'Portuguese'] + df_data.loc[cond, 'Olympics']
            print("{}: {}".format(cond, EffectSizeSingle().get_value(a, b)))

    def plot_bar_comparing_humans(self, auto_plot=False):
        """
        Bar chart with CI error bars comparing condition 1-3
        :param auto_plot:
        :return:
        """
        auto_plot=True
        data = { 'student': pd.read_pickle(self.ds_student.path_final_evaluation_aggregated),
                 'income': pd.read_pickle(self.ds_income.path_final_evaluation_aggregated),
                 'olympia': pd.read_pickle(self.ds_olympia.path_final_evaluation_aggregated),
        }
        fig = HumanComparisonBarChart().get_figure(data, feature_range=range(1, 10), conditions=[ERCondition.EXPERT, ERCondition.DOMAIN, ERCondition.RANDOM]) # , 1
        plotly.offline.plot(fig, auto_open=auto_plot, filename=self.path_human_comparison_plot)
        for dataset in data:
            path = "{}human-comparison_{}.json".format(self.path_human_comparison_data, dataset)
            data[dataset].to_json(path) # TODO: remove count, std, ... which is not visualised
        return fig

    def plot_bar_comparing_humans2(self):
        """
        Bar chart with CI error bars comparing condition 1-3
        :param auto_plot:
        :return:
        """
        data = self.get_evaluation_data()
        fig = HumanComparisonBarChart().get_figure(data, feature_range=range(1, 10), conditions=['Data Scientists', 'Domain Experts', 'Random']) # , 1
        plotly.offline.plot(fig, auto_open=True, filename=self.path_human_comparison_plot)
        # for dataset in data:
        #     path = "{}human-comparison_{}.json".format(self.path_human_comparison_data, dataset)
        #     data[dataset].to_json(path) # TODO: remove count, std, ... which is not visualised
        return fig

    def plot_bar_humans_vs_csfs(self, auto_plot=True, feature_range=range(1,10), plot_conditions=['KrowDD', 'Human'], recalc=False):
        """
        Bar chart comparing the combined condition (data scientsts + domain experts) with csfs
        change classifier 3 times (data in, data out, vis filename)
        :param auto_plot:
        :return:
        """
        datasets = ['income', 'student', 'olympia']
        if recalc:
            # plot_conditions=['KrowDD', 'Human', 'Random', 'Laypeople'] # for plotting
            data = { 'student': pd.DataFrame(pickle.load(open(self.ds_student.path_final_evaluation_aucs_nb, 'rb'))),
                             'income': pd.DataFrame(pickle.load(open(self.ds_income.path_final_evaluation_aucs_nb, 'rb'))),
                             'olympia': pd.DataFrame(pickle.load(open(self.ds_olympia.path_final_evaluation_aucs_nb, 'rb'))),
                    }
            for ds in data:
                data[ds].columns = [ERCondition.get_string_short(c) for c in data[ds].columns]

            def prepare_row(row):
                """
                Returns new row with two columns: combined experts and data scientists + cfs
                :param row:
                :return:
                """
                values_csfs = row['csfs']
                values_human = row['experts'] + row['domain']
                values_random = row['random']
                values_lay = row['lay']
                row_new = pd.Series({'KrowDD': values_csfs, 'Human': values_human, 'Random': values_random, 'Laypeople': values_lay})
                return row_new

            data_filtered = {ds_name: data[ds_name].loc[feature_range].apply(prepare_row, axis='columns') for ds_name in data}
            for d in data_filtered:
                data_filtered[d].to_json("{}{}_krowdd_vs_human_nb.json".format(self.path_csfs_vs_humans_data, d))
        else:
            data_filtered = {ds_name: pd.read_json("{}{}_krowdd_vs_human_nb.json".format(self.path_csfs_vs_humans_data, ds_name)).sort_index() for ds_name in datasets}
        # reduce conditions for plotting
        data_filtered = {ds_name: data_filtered[ds_name].loc[feature_range, plot_conditions] for ds_name in datasets }
        fig = CSFSVsHumansBarChart().get_figure(data=data_filtered, feature_range=range(1,10))
        plotly.offline.plot(fig, auto_open=True, filename=self.path_csfs_vs_humans_plot_mlp)

    def get_data_best_classifier(self, ds_names, feature_range):
        def get_best(all_data, n_feat, ds_name, classifiers):
            scores = {c: np.mean(all_data[c][ds_name]['KrowDD'][n_feat]) for c in classifiers}
            c_max = max(scores, key=scores.get)
            return c_max, scores[c_max]
        classifiers = ['nb', 'dt', 'mlp']
        all_data = {c: self.get_evaluation_data(c) for c in classifiers}

        data = {ds_name: {x: {'classifier': None, 'avg_auc': None} for x in feature_range} for ds_name in ds_names}
        # e.g. {student: {1: {classifier: mlp, avg_auc: 0.6}, 2: {classifier: dt, avg_auc: 0.61},...}, olympia: {...}}
        for ds_name in ds_names:
            for n_feat in feature_range:
                classifier, avg_auc = get_best(all_data, n_feat, ds_name, classifiers)
                data[ds_name][n_feat]['classifier'] = classifier
                data[ds_name][n_feat]['avg_auc'] = avg_auc
        return data

    def get_data_worst_classifier(self, ds_names, feature_range):
        # nearly a copy of get_data_best_classifier...
        def get_best(all_data, n_feat, ds_name, classifiers):
            scores = {c: np.mean(all_data[c][ds_name]['KrowDD'][n_feat]) for c in classifiers}
            c_max = min(scores, key=scores.get)
            return c_max, scores[c_max]
        classifiers = ['nb', 'dt', 'mlp']
        all_data = {c: self.get_evaluation_data(c) for c in classifiers}

        data = {ds_name: {x: {'classifier': None, 'avg_auc': None} for x in feature_range} for ds_name in ds_names}
        # e.g. {student: {1: {classifier: mlp, avg_auc: 0.6}, 2: {classifier: dt, avg_auc: 0.61},...}, olympia: {...}}
        for ds_name in ds_names:
            for n_feat in feature_range:
                classifier, avg_auc = get_best(all_data, n_feat, ds_name, classifiers)
                data[ds_name][n_feat]['classifier'] = classifier
                data[ds_name][n_feat]['avg_auc'] = avg_auc
        return data

    def get_data_classifiers(self, ds_names, feature_range, conditions):
        """

        :param ds_names:
        :param feature_range:
        :param conditions: list e.g. ['KrowDD'] or ['Data Scientists', 'Domain Experts']
        :return:
        """
        classifiers = ['dt', 'mlp']
        all_data = {c: self.get_evaluation_data(c) for c in classifiers}

        data = {ds_name: {x: {c: None} for c in classifiers for x in feature_range} for ds_name in ds_names}
        # e.g. {student: {1: {classifier: mlp, avg_auc: 0.6}, 2: {classifier: dt, avg_auc: 0.61},...}, olympia: {...}}
        for ds_name in ds_names:
            for n_feat in feature_range:
                for classifier in classifiers:
                    aucs = reduce(lambda a, b: a + b, [all_data[classifier][ds_name][cond][n_feat] for cond in conditions], [])
                    avg_auc = np.mean(aucs)
                    data[ds_name][n_feat][classifier] = avg_auc
        return data


    def plot_bar_humans_vs_csfs2(self, feature_range=range(1,10)):
        """
        Bar chart comparing the combined condition (data scientsts + domain experts) with csfs
        v2.0: Shows points for all classifiers
        :param auto_plot:
        :return:
        """
        data = self.get_evaluation_data(classifier='nb')
        def prepare_row(row):
                """
                Returns new row with two columns: combined experts and data scientists + cfs
                :param row:
                :return:
                """
                values_csfs = row['KrowDD']
                values_human = row['Domain Experts'] + row['Data Scientists']
                row_new = pd.Series({'KrowDD': values_csfs, 'Human': values_human})
                return row_new
        data_filtered = {ds_name: data[ds_name].loc[feature_range].apply(prepare_row, axis='columns') for ds_name in data}

        data_best_classifier = self.get_data_best_classifier(data.keys(), feature_range)  # data for best/worst classifier for each
        data_worst_classifier = self.get_data_worst_classifier(data.keys(), feature_range)  # data for best/worst classifier for each
        data_classifiers_krowdd = self.get_data_classifiers(data.keys(), feature_range, conditions=['KrowDD'])
        data_classifiers_human = self.get_data_classifiers(data.keys(), feature_range, conditions=['Data Scientists', 'Domain Experts'])

        fig = CSFSVsHumansBarChart().get_figure(data=data_filtered, data_classifiers_krowdd=data_classifiers_krowdd, data_classifiers_human=data_classifiers_human, data_best_classifier=data_best_classifier, data_worst_classifier=data_worst_classifier, feature_range=range(1, 10))
        plotly.offline.plot(fig, auto_open=True, filename=self.path_csfs_vs_humans_plot_with_classifiers)

    def plot_bar_humans_vs_csfs3(self, feature_range=range(1,10)):
        """
        Bar chart comparing the combined condition (data scientsts + domain experts) with csfs
        v3.0: Shows best/worst
        :param auto_plot:
        :return:
        """
        data = self.get_evaluation_data(classifier='nb')
        def prepare_row(row):
                """
                Returns new row with two columns: combined experts and data scientists + cfs
                :param row:
                :return:
                """
                values_csfs = row['KrowDD']
                values_human = row['Domain Experts'] + row['Data Scientists']
                values_best = row['Best']
                values_worst = row['Worst']
                row_new = pd.Series({'KrowDD': values_csfs, 'Human': values_human, 'Best': values_best, 'Worst': values_worst,
                                     'Random': row['Random']})
                return row_new
        data_filtered = {ds_name: data[ds_name].loc[feature_range].apply(prepare_row, axis='columns') for ds_name in data}

        fig = CSFSVsHumansBarChart3().get_figure(data=data_filtered, feature_range=feature_range)
        plotly.offline.plot(fig, auto_open=True, filename=self.path_csfs_vs_humans_plot_with_classifiers)

    def table_human_vs_csfs(self):
        """
        :pre: json data already saved in plot above
        :return:
        """
        datasets = ['student', 'income', 'olympia']
        data = { d: pd.read_json("{}{}_krowdd_vs_human.json".format(self.path_csfs_vs_humans_data, d)).sort_index() for d in datasets}
        feature_range = range(1,10)
        df_result = pd.DataFrame(index=feature_range)
        for d in datasets:
            df = data[d]
            series = EffectSizeTable(df, feature_range=feature_range).get_result_series(dataset_name=d, condition_better='KrowDD', condition_other='Human')
            df_result[d] = series
        df_result.columns=['Student', 'Income', 'Olympics']
        print(df_result.to_latex(escape=False))

    def table_human_vs_csfs2(self):
        """
        :pre: json data already saved in plot above
        :return:
        """
        data = self.get_evaluation_data()
        feature_range = range(1,10)
        def prepare_row(row): # copy from plot function
                """
                Returns new row with two columns: combined experts and data scientists + cfs
                :param row:
                :return:
                """
                values_csfs = row['KrowDD']
                values_human = row['Domain Experts'] + row['Data Scientists']
                row_new = pd.Series({'KrowDD': values_csfs, 'Human': values_human})
                return row_new
        data_filtered = {ds_name: data[ds_name].loc[feature_range].apply(prepare_row, axis='columns') for ds_name in data}
        df_result = pd.DataFrame(index=feature_range)
        for d in data_filtered:
            df = data_filtered[d]
            series = EffectSizeTable(df, feature_range=feature_range).get_result_series(dataset_name=d, condition_better='KrowDD', condition_other='Human')
            df_result[d] = series
        print(df_result.to_latex(escape=False))

    def table_lay_vs_csfs(self):
        data = self.get_evaluation_data()
        feature_range = range(1,10)
        df_result = pd.DataFrame(index=feature_range)
        for d in data:
            df = data[d]
            series = EffectSizeTable(df, feature_range=feature_range, conditions=['KrowDD', 'Laypeople']).get_result_series(dataset_name=d, condition_better='KrowDD', condition_other='Laypeople')
            df_result[d] = series
        print(df_result.to_latex(escape=False))

    def move_and_rename_auc_for_all_conditions(self):
        """
        Moves complete table with aucs for all conditions to paper folder and renames columns to paper style.
        :return:
        """
        #TODO: REDO when decision for method name has happened + adjust dataset names
        names = dict(
            best=ERCondition.get_string_paper(7),
            domain=ERCondition.get_string_paper(2),
            experts=ERCondition.get_string_paper(3),
            lay=ERCondition.get_string_paper(1),
            random=ERCondition.get_string_paper(5),
            worst=ERCondition.get_string_paper(8),
            csfs=ERCondition.get_string_paper(4),
        )

        data = { 'student': pd.read_json(self.ds_student.path_auc_all_conditions),
                         'income': pd.read_json(self.ds_income.path_auc_all_conditions),
                         'olympia': pd.read_json(self.ds_olympia.path_auc_all_conditions),
                }
        for dataset in data:
            df = data[dataset]
            columns_new = [names[c] for c in df.columns]
            df.columns = columns_new
            path_out_csv = "{}{}_auc_all_conditions.csv".format(self.path_auc_all_conditions, dataset)
            df.to_csv(path_out_csv)
            path_out_json = "{}{}_auc_all_conditions.json".format(self.path_auc_all_conditions, dataset)
            df.to_csv(path_out_json)

    def single_humans_performance(self):
        feature_range = range(1, 10)
        df_evaluation_result = pd.read_csv('final_evaluation/private_conditions1-3_result.csv', header=None, names=['id', 'dataset_name', 'condition', 'name', 'token', 'comment', 'ip', 'date'])
        datasets = {
            'student': self.ds_student,
            'income': self.ds_income,
            'olympia': self.ds_olympia
        }
        def add_auc_column(row):
            exp = datasets[row.dataset_name]
            token = row.token
            df_evaluation_base = pd.read_csv(exp.path_budget_evaluation_base)
            df_cleaned_bin = pd.read_csv(exp.path_bin)
            target = exp.target
            df_features_ranked = ERParser(df_evaluation_base).get_ordered_features(token)
            evaluator = AUCForOrderedFeaturesCalculator(df_features_ranked, df_cleaned_bin, target)
            df_aucs = evaluator.get_auc_for_nofeatures_range(feature_range) # df with one col: AUC and index= cost
            aucs = {no_features: df_aucs.loc[no_features, 'auc'] for no_features in df_aucs.index}
            row['aucs'] = aucs
            return row
        # df_evaluation_result = df_evaluation_result.loc[:10]
        df_evaluation_result = df_evaluation_result.apply(add_auc_column, axis=1)
        df_single_human_performance = df_evaluation_result.drop(['token', 'id'], axis=1)
        # print(tabulate(df_single_human_performance, headers='keys'))
        df_single_human_performance.to_json(self.path_single_human_performance_data)

    def data_scientist_performance(self):
        df_participants = pd.read_csv(self.path_data_scientists)
        df_single_humans = pd.read_json(self.path_single_human_performance_data)
        # print(tabulate(df_single_humans.head(), headers='keys'))
        df_single_humans = df_single_humans[df_single_humans['condition'] == ERCondition.EXPERT]
        # print(tabulate(df_participants.head(), headers='keys'))
        df_joined = df_participants.merge(df_single_humans, left_on='Username', right_on='name')
        df_joined = df_joined.drop(['Username', 'Country', 'When is your birthday?', 'When did you start working in data science? If you are not sure about the exact date just select a random day in the year you became a data scientist.', 'Please explain your choice in a few words', 'Are you interested in learning about the results of our research?', 'Comments', 'comment', 'ip', 'date', 'name', 'condition'], axis=1)
        print(tabulate(df_joined, headers='keys'))
        df_joined.to_json(self.path_data_scientists_performance)

    def chosen_features_ig(self):
        """
        Lists chosen binary features with their IG. Saves one CSV for each dataset
        :return:
        """
        def get_df_chosen_features(exp):
            df_meta = pd.read_csv(exp.path_meta, index_col=0)
            df_meta = df_meta[['IG']]
            features = pd.read_csv(exp.path_budget_evaluation_base).Feature
            df_chosen = df_meta.loc[features]
            df_chosen = df_chosen.sort_values('IG', ascending=False)
            return df_chosen

        for dataset in self.datasets:
            df_chosen_features = get_df_chosen_features(self.datasets[dataset])
            df_chosen_features.to_csv("{}{}_chosen_features_ig.csv".format(self.path_chosen_features_ig, dataset))

    def compare_classifiers(self):
        """
        Compares Naive Bayes with Decision Tree and Multilayer perceptron
        - t-test
        if difference:
        - correlation of conditions
        :return:
        """
        for dataset in self.datasets:
            feature_slice = 9
            conditions = ['csfs', 'domain', 'experts', 'lay', 'random']

            scores = {
                'MLP': pd.DataFrame(pickle.load(open(self.datasets[dataset].path_final_evaluation_aucs_mlp, 'rb'))),
                'DT':  pd.DataFrame(pickle.load(open(self.datasets[dataset].path_final_evaluation_aucs_dt, 'rb'))),
                'NB':  pd.DataFrame(pickle.load(open(self.datasets[dataset].path_final_evaluation_aucs_nb, 'rb'))),
            }
            for c in scores:
                scores[c].columns = [ERCondition.get_string_short(c) for c in scores[c].columns]

            classifiers = [c for c in scores]
            combinations = list(itertools.combinations(classifiers, 2))
            # print(scores['MLP'])
            # print(scores['DT'])
            # exit()

            df_t = pd.DataFrame(columns=conditions, index=["{} vs. {}".format(comb[0], comb[1]) for comb in combinations])
            df_corr = pd.DataFrame(columns=conditions, index=["{} vs. {}".format(comb[0], comb[1]) for comb in combinations])
            # print(df)
            # exit()
            for condition in conditions:
                for combination in combinations:
                    print(combination)
                    a = scores[combination[0]].loc[feature_slice, condition]
                    b = scores[combination[1]].loc[feature_slice, condition]
                    # print(a)
                    # print(b)
                    eff_size = EffectSizeSingle().get_value(a, b)
                    # print(a,b,eff_size)
                    df_t.loc["{} vs. {}".format(combination[0], combination[1]), condition] = eff_size
                    df_corr.loc["{} vs. {}".format(combination[0], combination[1]), condition] = EffectSizeSingle().get_correlation(a, b)

            print(dataset, "Welch's t-test + Hedges' g" )
            print(tabulate(df_t, headers='keys'))
            print(dataset, "Spearman")
            print(tabulate(df_corr, headers='keys'))
            print('--')


    def compare_classifiers_vis(self):
        """
        Compares Naive Bayes with Decision Tree and Multilayer perceptron
        - t-test
        if difference:
        - correlation of conditions
        :return:
        """
        for dataset in self.datasets:
            scores = {
                'MLP': pd.DataFrame(pickle.load(open(self.datasets[dataset].path_final_evaluation_aucs_mlp, 'rb'))),
                'DT':  pd.DataFrame(pickle.load(open(self.datasets[dataset].path_final_evaluation_aucs_dt, 'rb'))),
                'NB':  pd.DataFrame(pickle.load(open(self.datasets[dataset].path_final_evaluation_aucs_nb, 'rb'))),
            }
            for c in scores:
                scores[c].columns = [ERCondition.get_string_short(c) for c in scores[c].columns]

            conditions = ['lay', 'domain', 'experts', 'csfs']
            for c in conditions:
                # print(scores['DT'])
                # exit()
                data = {
                    s:scores[s][c] for s in scores
                }
                df = pd.DataFrame(data)
                fig = ClassifiersComparisonBarChart().get_figure(df, range(1,10), c)
                filename = "{}{}_{}_classifier-performance.html".format(self.path_compare_classifiers, dataset, c)
                plotly.offline.plot(fig, filename=filename, auto_open=False)

    def save_data_for_paper(self):
        for dataset in self.datasets:
            df = pd.DataFrame(pickle.load(open(self.datasets[dataset].path_final_evaluation_aucs_nb, 'rb')))
            df.columns = [ERCondition.get_string_paper(c) for c in df.columns]
            path = "{}{}_evaluated_nb.json".format(self.path_data, dataset)
            df.to_json(path)

    def tmp(self):
        for dataset in self.datasets:
            path = "{}{}_evaluated_nb.json".format(self.path_data, dataset)
            df = pd.read_json(path)
            df.columns = [c[:1].upper()+c[1:] for c in df.columns]
            # df['Random'] = df['Random'].apply(lambda x: np.random.choice(x, 19, replace=False))
            print(list(df.columns))
            df.to_json("{}{}_evaluated_nb.json".format(self.path_data, dataset))

    def copy_bin_datasets(self):
        for dataset in self.datasets:
            df = pd.read_csv(self.datasets[dataset].path_bin)
            # remove features that have not been used in evaluation
            df_questions = pd.read_csv(self.datasets[dataset].path_questions, header=None)
            features = get_features_from_questions(self.datasets[dataset].path_questions, remove_cond=True)
            features.append(self.datasets[dataset].target)
            df = df[features]
            path = "{}{}_bin.csv".format(self.path_data, dataset)
            df.to_csv(path, index=False)
            # exit()

    def human_comparison_table(self):
        # effect size table for human comparison. previously in experiments
        data = self.get_evaluation_data()
        # print(data)
        conditions = ['Domain Experts', 'Data Scientists', 'Random']
        for dataset in data:
            print(dataset)
            df = data[dataset]
            table = EffectSizeMatrix(df, conditions, remove_null=True, rename_columns=False)
            latex = table.get_latex()
            print(latex)





def run():
    experiment = MetaExperiment()
    # experiment.final_evaluation_combine_all()
    # experiment.plot_humans_vs_actual_all_plot()
    # experiment.plot_humans_vs_actual_all_plot2()
    # experiment.plot_no_answers_vs_delta()
    # experiment.table_kahneman()
    # experiment.plot_bar_comparing_humans()
    # experiment.table_human_vs_csfs()
    # experiment.table_human_vs_csfs2()
    # experiment.table_lay_vs_csfs()
    # experiment.move_and_rename_auc_for_all_conditions()
    # experiment.single_humans_performance()
    # experiment.data_scientist_performance()
    # experiment.chosen_features_ig()

    # experiment.compare_classifiers()
    # experiment.compare_classifiers_vis()
    # experiment.plot_bar_humans_vs_csfs2()
    experiment.plot_bar_humans_vs_csfs3()
    # experiment.save_data_for_paper()
    # experiment.plot_bar_comparing_humans2()
    # experiment.tmp()
    # experiment.copy_bin_datasets()
    # experiment.human_comparison_table()

if __name__ == '__main__':
    run()