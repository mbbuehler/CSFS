import pickle
from collections import Counter

from CSFSSelector import CSFSRandomSelector, CSFSBestActualSelector, CSFSBestUncertainSelector

from feature_subset_comparison2 import AUCComparator
import numpy as np
import matplotlib.pyplot as plt

class CSFSEvaluator:

    def __init__(self, df, target):
        self.comparator = AUCComparator(df, target, fast=False, n_folds=10)
        self.df = df
        self.target = target
        # this one will return deterministic results -> create here for better performance
        # self.best_selector = CSFSBestActualSelector(self.df, self.target)
        # self.fix_std = fix_std

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

    def evaluate_best(self, N_features, best_selector): # always the same, don't need many samples
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

    def evaluate_features(self, selected_features):
        """

        :param features:
        :return:
        """
        auc = self._get_mean_auc_score(selected_features)
        return auc
