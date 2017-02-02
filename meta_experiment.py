import pandas as pd
import numpy as np

import plotly
from csfs_visualisations import HumanVsActualBarChart
from experiment_income import ExperimentIncome
from experiment_olympia import ExperimentOlympia
from experiment_student_por import ExperimentStudent


class MetaExperiment:
    def __init__(self):
        self.path_final_evaluation_combined = 'final_evaluation/combined.csv'
        self.path_upwork_participants = 'participants.csv'
        self.path_statistical_comparison = 'final_evaluation/comparisons/' # e.g. + 'student_1-vs-2'

        self.ds_student = ExperimentStudent('student', 2, 'experiment2_por')
        self.ds_income = ExperimentIncome('income', 1, 'experiment1')
        self.ds_olympia = ExperimentOlympia('olympia', 4, 'experiment2-4_all')

    def final_evaluation_combine_all(self):

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
        max_answer_count = max(list(df_student.index) + list(df_olympia.index) + list(df_income.index))
        range_answers = range(1, max_answer_count)

        def get_df_normalised(df):
            print(df.head())

        get_df_normalised(df_student)
        exit()

        def normalise(x, min, max):
            print(x, min, max)
            if x == min == max:
                z = -1
            else:
                z = (x - min) / (max - min)
            # assert 0 <= z <= 1

            print(z)
            return z

        def get_series_relative_means(list_df, no_answers):
            print(list_df[0].head())
            for df in list_df:
                worst = df.loc[no_answers, 'worst']
                best = df.loc[no_answers, 'best']
                print([normalise(v, worst, best) for v in list(df.loc[no_answers, 'domain'])])
                # exit()
                df.loc[no_answers, 'domain'] = [normalise(v, worst, best) for v in list(df.loc[no_answers, 'domain'])]
                exit()
                # df.loc[no_answers, 'experts'] = normalise(df.loc[no_answers, 'experts'], df.loc[no_answers, 'worst'], df.loc[no_answers, 'best'])
                # df.loc[no_answers, 'lay'] = normalise(df.loc[no_answers, 'lay'], df.loc[no_answers, 'worst'], df.loc[no_answers, 'best'])
            print(list_df[0].head())
            exit()

            # list_df = [df for df in list_df if no_answers in df.index]
            rel_means_domain = [normalise(np.mean(df.loc[no_answers, 'domain']), df.loc[no_answers, 'worst'], df.loc[no_answers, 'best']) for df in list_df if no_answers in df.index]
            rel_means_experts = [normalise(np.mean(df.loc[no_answers, 'experts']), df.loc[no_answers, 'worst'], df.loc[no_answers, 'best']) for df in list_df if no_answers in df.index]
            rel_means_lay = [normalise(np.mean(df.loc[no_answers, 'lay']), df.loc[no_answers, 'worst'], df.loc[no_answers, 'best']) for df in list_df if no_answers in df.index]

            # normalise returns -1 if x, min and max are equal. ignore these values. nan was not properly recognised, so use -1
            rel_means_domain = [mean for mean in rel_means_domain if mean != -1]
            rel_means_experts = [mean for mean in rel_means_experts if mean != -1]

            row_rel = pd.Series({
                'Domain Experts': np.mean(rel_means_domain),
                'Data Science Experts': np.mean(rel_means_experts),
                'Laymen': np.mean(rel_means_lay),
            })
            return row_rel

        df_result = pd.DataFrame(index=range_answers, columns=['Data Science Experts', 'Domain Experts', 'Laymen'])
        for no_answers in range_answers:
            # print()
            df_result.loc[no_answers] = get_series_relative_means(df_all, no_answers)
        # df_result contains the normalised averaged values for domain experts and datascience experts for all three datasets. the normalisation took place for each dataset, not for the aggregated data.
        fig = HumanVsActualBarChart().get_figure(df_result)
        plotly.offline.plot(fig, auto_open=True)

    def plot_no_answers_vs_delta(self):
        """
        Combined Boxplot for number of answers versus combined error for all three conditions.
        :return:
        """
        df = pd.read_json('../comparison/humans_vs_actual_auc.json').sort_index()
        df_student = pd.read_json(self.ds_student.path_humans_vs_actual_auc).sort_index()
        df_income = pd.read_json(self.ds_income.path_humans_vs_actual_auc).sort_index()
        df_olympia = pd.read_json(self.ds_olympia.path_humans_vs_actual_auc).sort_index()
        df_all = [df_student, df_income, df_olympia]
        answer_range = range(1, 17)

        # combine mean list for all three datasets and then create a box for all number of answers








def run():
    experiment = MetaExperiment()
    # experiment.final_evaluation_combine_all()
    experiment.plot_humans_vs_actual_all_plot()


if __name__ == '__main__':
    run()