from experiment_income import ExperimentIncome
from experiment_olympia import ExperimentOlympia
from experiment_student_por import ExperimentStudent


def run_all():
    exp_student = ExperimentStudent('student', 2, 'experiment2_por')
    exp_olympia = ExperimentOlympia('olympia', 4, 'experiment2-4_all')
    exp_income = ExperimentIncome('income', 1, 'experiment1')

    experiments = [exp_student, exp_income, exp_olympia]

    for exp in experiments:
        exp.run()

if __name__=='__main__':
    run_all()