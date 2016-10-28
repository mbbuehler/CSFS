import pickle
from collections import Counter

from CSFSSelector import CSFSRandomSelector, CSFSBestActualSelector, CSFSBestUncertainSelector

from feature_subset_comparison2 import AUCComparator
import numpy as np
import matplotlib.pyplot as plt

class CSFSEvaluator:

    def __init__(self, df, target, fix_std=None ):
        self.comparator = AUCComparator(df, target, fast=False, n_folds=2)
        self.df = df
        self.target = target
        # this one will return deterministic results -> create here for better performance
        self.best_selector = CSFSBestActualSelector(self.df, self.target)
        self.fix_std = fix_std

    def _get_mean_auc_score(self, features):
        return self.comparator.get_mean_score(features)

    def evaluate_noisy(self, N_features, N_samples, best_noisy_selector):
        aucs = {'best_noisy': []}
        selected_features = list()
        for i in range(N_samples):
            best_noisy_f = best_noisy_selector.select(N_features)
            selected_features += best_noisy_f
            aucs['best_noisy'].append(self._get_mean_auc_score(best_noisy_f))
        aucs['best_noisy_features_count'] = Counter(selected_features) # Counter({'blue': 3, 'red': 2, 'yellow': 1})
        return aucs

    def evaluate_noisy_mean(self, N_features, N_samples, all_features_noisy_selector):
        aucs = {'best_noisy_mean': []}
        selected_features = list()
        for i in range(N_samples):
            best_noisy_f = all_features_noisy_selector.select(N_features)
            selected_features += best_noisy_f
            aucs['best_noisy_mean'].append(self._get_mean_auc_score(best_noisy_f))
        aucs['best_noisy_mean_features_count'] = Counter(selected_features) # Counter({'blue': 3, 'red': 2, 'yellow': 1})
        return aucs

    def evaluate_best(self, N_features, N_samples, best_selector): # always the same, don't need many samples
        aucs = {'best': []}
        selected_features = best_selector.select(N_features)
        aucs['best'].append(self._get_mean_auc_score(selected_features))
        aucs['best_features_count'] = Counter(selected_features) # Counter({'blue': 3, 'red': 2, 'yellow': 1})
        return aucs

    def evaluate_random(self, N_features, N_samples, random_selector):
        aucs = {'random': []}
        selected_features = list()
        for i in range(N_samples):
            best_f = random_selector.select(N_features)
            selected_features += best_f
            aucs['random'].append(self._get_mean_auc_score(best_f))
        aucs['random_features_count'] = Counter(selected_features) # Counter({'blue': 3, 'red': 2, 'yellow': 1})
        return aucs


    def evaluate(self, N_features=5, N_samples=100):
        """

        :param df:
        :param target:
        :param N_features:
        :param N_samples:
        :return: df with auc scores
        """
        random_selector = CSFSRandomSelector(self.df, self.target)
        print('> random selector ok')
        # best_selector = CSFSBestActualSelector(self.df, self.target)
        # print('> best selector ok')
        best_noisy_selector = CSFSBestUncertainSelector(self.df, self.target, fix_std=self.fix_std)
        print('> best noisy selector ok')

        aucs = {'random': [], 'best': [], 'best_noisy': []}
        for i in range(N_samples):
            print('> processing sample {} from {} samples'.format(i, N_samples))
            random_f = random_selector.select(N_features)
            best_f = self.best_selector.select(N_features)
            best_noisy_f = best_noisy_selector.select(N_features)

            aucs['random'].append(self._get_mean_auc_score(random_f))
            aucs['best'].append(self._get_mean_auc_score(best_f))
            aucs['best_noisy'].append(self._get_mean_auc_score(best_noisy_f))
        return aucs

    def plot(self, auc_scores, info, show=False):
        """

        :param auc_scores: df mit
        :param info: dict with at least 'dataset', 'N_features' and 'N_samples': {'dataset': dataset_name, 'N_features': N_features, 'N_samples': N_samples}
        :param show: show plot boolean
        best  best_noisy    random
    0   0.687500    0.718750  0.687500
    1   0.656250    0.718750  0.718750

        :return:
        """
        scores_count = len(auc_scores['best'])
        legend_list = []
        for c in auc_scores:
            mean = np.mean(auc_scores[c])
            std = np.std(auc_scores[c])
            plt.plot(auc_scores[c])
            plt.hold(True)
            plt.plot([mean]*scores_count)
            legend_list.append("{} (std={:.4f})".format(c, std))
            legend_list.append("mean {} ({:.2f})".format(c, mean))
        plt.title('Dataset: {} (using {} features)'.format(info['dataset'], info['N_features']))
        plt.legend(legend_list)
        plt.xlabel('# samples (total: {})'.format(scores_count))
        plt.ylabel('AUC')
        fig1 = plt.gcf()
        filepath = '{}/{}features_{}samples_{:.3f}std'.format(info['dataset'], info['N_features'], scores_count, info['std'])
        fig1.savefig('plots/{}.png'.format(filepath), dpi=100)
        pickle.dump(auc_scores, open("pickle-dumps/{}.pickle".format(filepath), 'wb'))

        if show:
            plt.show()
        plt.draw()
        plt.hold(False)
