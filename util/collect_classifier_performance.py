import json
import pickle


class ERCondition:
    LAYPERSON = 1 # AMT Turkers
    DOMAIN = 2 # e.g. Teachers
    EXPERT = 3 # Upwork
    CSFS = 4
    RANDOM = 5
    ACTUAL = 6
    BEST = 7
    WORST = 8
    HUMAN = 9

    NAMES_SHORT = {
        LAYPERSON: 'lay',
        DOMAIN: 'domain',
        EXPERT: 'experts',
        CSFS: 'csfs',
        RANDOM: 'random',
    }
    NAMES = {
        LAYPERSON: 'lay',
        DOMAIN: 'domain expert',
        EXPERT: 'data scientist',
        CSFS: 'CSFS',
        RANDOM: 'random',
        ACTUAL: 'actual',
        HUMAN: 'human experts'
    }

    PAPER_NAMES = {
        LAYPERSON: 'Laypeople',
        DOMAIN: 'Domain Experts',
        EXPERT: 'Data Scientists',
        CSFS: 'KrowDD',
        RANDOM: 'Random',
        ACTUAL: 'actual',
        BEST: 'Best',
        WORST: 'Worst',
        HUMAN: 'Human Experts'
    }

    @staticmethod
    def get_all():
        return [ERCondition.LAYPERSON, ERCondition.DOMAIN, ERCondition.EXPERT, ERCondition.CSFS, ERCondition.RANDOM, ERCondition.ACTUAL]

    @staticmethod
    def get_string(condition):
        if condition in ERCondition.NAMES:
            return ERCondition.NAMES[condition]
        return 'n.a.'

    @staticmethod
    def get_string_paper(condition):
        if condition in ERCondition.PAPER_NAMES:
            return ERCondition.PAPER_NAMES[condition]
        print(condition)
        return 'n.a.'

    @staticmethod
    def get_string_short(condition):
        if condition in ERCondition.NAMES_SHORT:
            return ERCondition.NAMES_SHORT[condition]
        return 'n.a.'




classifiers = ['dt', 'mlp']
for c in classifiers:
    path_in = 'datasets/student/evaluation/final_evaluation_aucs_{}.pickle'.format(c)
    data_in = pickle.load(open(path_in, 'rb'))
    data_out = {ERCondition.get_string_paper(c): data_in[c] for c in data_in}
    path_out = 'paper_plots-and-data/datasets/Student_evaluated_{}.json'.format(c)
    json.dump(data_out, open(path_out, 'w'))
