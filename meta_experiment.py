import json
import math
import pandas as pd
import numpy as np

import plotly
from tabulate import tabulate

from application.EvaluationRanking import ERCondition
from csfs_visualisations import HumanVsActualBarChart, AnswerDeltaVisualiserBox, HumanComparisonBarChart, \
    CSFSVsHumansBarChart
from experiment_income import ExperimentIncome
from experiment_olympia import ExperimentOlympia
from experiment_student_por import ExperimentStudent
from table_effect_size import EffectSizeTable


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

        self.path_csfs_vs_humans_plot = 'paper_plots-and-data/krowdd_vs_humans/krowdd_vs_humans.html'
        self.path_csfs_vs_humans_data = 'paper_plots-and-data/krowdd_vs_humans/'

        self.ds_student = ExperimentStudent('student', 2, 'experiment2_por')
        self.ds_income = ExperimentIncome('income', 1, 'experiment1')
        self.ds_olympia = ExperimentOlympia('olympia', 4, 'experiment2-4_all')



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
        df_student = pd.read_json(self.ds_student.path_humans_vs_actual_auc).sort_index()
        df_income = pd.read_json(self.ds_income.path_humans_vs_actual_auc).sort_index()
        df_olympia = pd.read_json(self.ds_olympia.path_humans_vs_actual_auc).sort_index()
        df_all = [df_student, df_income, df_olympia]
        max_feature_count = max(list(df_student.index) + list(df_olympia.index) + list(df_income.index))
        range_features = range(1, max_feature_count+1)
        conditions = ['domain', 'experts', 'lay', 'random']

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
        print(df_result.head())
        df_result.columns = ['Domain Experts', 'Data Science Experts', 'Laypeople', 'Random']
        df_result.to_json(self.path_human_vs_actual_data)

        fig = HumanVsActualBarChart().get_figure(df_result, feature_range=range(1,10))
        plotly.offline.plot(fig, auto_open=True, image='png', filename=self.path_human_vs_actual_barchart)
        fig = HumanVsActualBarChart().get_histograms(df_result)
        plotly.offline.plot(fig, auto_open=True, image='png', filename=self.path_human_vs_actual_histogram)

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

        data = dict()
        for no_answers in range_answers:
            data[no_answers] = list()
            for df in df_all:
                for cond in conditions:
                    data[no_answers] += df.loc[no_answers, cond]
        df_no_answers_vs_delta = pd.DataFrame(data) # columns are no_answers
        df_no_answers_vs_delta.to_json(self.path_plot_no_answers_vs_delta_data)
        # 'Number of Answers versus Actual Data (19 Repetitions)'
        fig = AnswerDeltaVisualiserBox(title="").get_figure(df_no_answers_vs_delta)
        plotly.offline.plot(fig, auto_open=True, filename=self.path_plot_no_answers_vs_delta_html, image='png', image_filename=self.path_plot_no_answers_vs_delta_png)

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
        fig = HumanComparisonBarChart().get_figure(data, feature_range=range(1, 10), conditions=[1,2,3])
        plotly.offline.plot(fig, auto_open=auto_plot, filename=self.path_human_comparison_plot)
        for dataset in data:
            path = "{}human-comparison_{}.json".format(self.path_human_comparison_data, dataset)
            data[dataset].to_json(path) # TODO: remove count, std, ... which is not visualised
        return fig

    def plot_bar_humans_vs_csfs(self, auto_plot=True, feature_range=range(1,10)):
        """
        Bar chart comparing the combined condition (data scientsts + domain experts) with csfs
        :param auto_plot:
        :return:
        """
        data = { 'student': pd.read_json(self.ds_student.path_auc_all_conditions).sort_index(),
                         'income': pd.read_json(self.ds_income.path_auc_all_conditions).sort_index(),
                         'olympia': pd.read_json(self.ds_olympia.path_auc_all_conditions).sort_index(),
                }

        def prepare_row(row):
            """
            Returns new row with two columns: combined experts and data scientists + cfs
            :param row:
            :return:
            """
            values_csfs = row['csfs']
            values_human = row['experts'] + row['domain']
            # values_random = row['random']
            row_new = pd.Series({'KrowDD': values_csfs, 'Human': values_human}) #, 'Random': values_random})
            return row_new

        data_filtered = {ds_name: data[ds_name].loc[feature_range].apply(prepare_row, axis='columns') for ds_name in data}
        fig = CSFSVsHumansBarChart().get_figure(data=data_filtered, feature_range=range(1,10))
        plotly.offline.plot(fig, auto_open=True, filename=self.path_csfs_vs_humans_plot)
        for d in data_filtered:
            data_filtered[d].to_json("{}{}_krowdd_vs_human.json".format(self.path_csfs_vs_humans_data, d))

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
            series = EffectSizeTable(df, feature_range=feature_range).get_result_series(dataset_name=d)
            df_result[d] = series
        df_result.columns=['Portuguese', 'Income', 'Olympics']
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

def run():
    experiment = MetaExperiment()
    # experiment.final_evaluation_combine_all()
    # experiment.plot_humans_vs_actual_all_plot()
    # experiment.plot_no_answers_vs_delta()
    # experiment.plot_bar_comparing_humans()
    # experiment.plot_bar_humans_vs_csfs()
    experiment.table_human_vs_csfs()
    # experiment.move_and_rename_auc_for_all_conditions()


if __name__ == '__main__':
    run()