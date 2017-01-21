import pandas as pd

from experiment_income import ExperimentIncome
from experiment_olympia import ExperimentOlympia
from experiment_student_por import ExperimentStudent


class MetaExperiment:
    def __init__(self):
        self.path_final_evaluation_combined = 'final_evaluation/combined.csv'
        self.path_upwork_participants = 'participants.csv'

    def final_evaluation_combine_all(self):
        student = ExperimentStudent('student', 2, 'experiment2_por')
        income = ExperimentIncome('income', 1, 'experiment1')
        olympia = ExperimentOlympia('olympia', 4, 'experiment2-4_all')

        df_student = pd.read_csv(student.path_final_evaluation_combined)
        df_income = pd.read_csv(income.path_final_evaluation_combined)
        df_olympia = pd.read_csv(olympia.path_final_evaluation_combined)

        df_combined_all = pd.concat([df_student, df_income, df_olympia])

        # df_participants = pd.read_csv()
        df_combined_all.to_csv(self.path_final_evaluation_combined, index=False)




def run():
    MetaExperiment().final_evaluation_combine_all()

if __name__ == '__main__':
    run()