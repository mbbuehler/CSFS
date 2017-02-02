import os
from joblib import Parallel, delayed

from experiment_income import ExperimentIncome
from experiment_olympia import ExperimentOlympia
from experiment_student_por import ExperimentStudent


def run_all():
    exp_student = ExperimentStudent('student', 2, 'experiment2_por')
    exp_olympia = ExperimentOlympia('olympia', 4, 'experiment2-4_all')
    exp_income = ExperimentIncome('income', 1, 'experiment1')

    experiments = [exp_student, exp_income, exp_olympia]


    Parallel(n_jobs=len(experiments))(delayed(run)(exp) for exp in experiments)
    # for exp in experiments:
    #     print('> running experiment', exp.dataset_name)
    #     exp.run()

def run(exp):
    print('> running dataset', exp.dataset_name)
    exp.run()
    print('> dataset done:', exp.dataset_name)

if __name__=='__main__':
    os.chdir('..')
    run_all()