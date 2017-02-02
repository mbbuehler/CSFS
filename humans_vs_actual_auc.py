import itertools

from joblib import Parallel, delayed

from CSFSEvaluator import CSFSEvaluator


class FeatureRankerAUC:
    """
    Uses AUC (not IG) to rank features. sequentially adds best / worst feature
    """
    def __init__(self, df_bin, target, features_all):
        self.df_bin = df_bin
        self.target = target
        self.features_all = features_all

    def find_best_feature(self, features_fix, reverse): #TODO: rename to generic (it can also be used for worst features by setting reverse=True)
        """
        Finds feature f that maximises auc calculate by features_fix + [f]
        :param features_fix: list(str)
        :return: str, float
        """
        r = {f: CSFSEvaluator(self.df_bin, self.target).evaluate_features(features_fix+[f]) for f in self.features_all if f not in features_fix}
        f_sorted = sorted(r, key=r.__getitem__, reverse=reverse)
        best_feature = f_sorted[-1]
        auc = r[best_feature]
        return best_feature, auc

    def get_ranked(self, reverse=False, return_features=False):
        result_auc = dict()
        features_fix = list()
        for no_features in range(1, len(self.features_all)+1):
            best_feature, auc_best = self.find_best_feature(features_fix, reverse=reverse)
            features_fix.append(best_feature)
            result_auc[no_features] = auc_best
        features_ranked = features_fix
        no_features_vs_auc = result_auc
        if return_features:
            return features_ranked, no_features_vs_auc
        return no_features_vs_auc

class FeatureCombinationCalculator:
    """
    Uses AUC (not IG) to rank features. Tries all permutations and takes n features
    """
    def __init__(self, df_bin, target, features_all):
        self.df_bin = df_bin
        self.target = target
        self.features_all = features_all

    def get_best_auc(self, feature_range):
        return self.get_aucs_for_feature_range(feature_range, reverse=False)

    def get_worst_auc(self, feature_range):
        return self.get_aucs_for_feature_range(feature_range, reverse=True)

    def get_max_auc(self, no_features, reverse=False):
        print('no features:', no_features)
        combinations = [list(c) for c in itertools.combinations(self.features_all, no_features)]
        evaluator = CSFSEvaluator(self.df_bin, self.target)
        aucs = Parallel(n_jobs=8)(delayed(evaluator.evaluate_features)(features) for features in combinations)
        if reverse:
            return min(aucs)
        return max(aucs)

    def get_aucs_for_feature_range(self, feature_range, reverse=False):
        result = dict()
        for no_features in feature_range:
            max_auc = self.get_max_auc(no_features, reverse=reverse)
            result[no_features] = max_auc
        return result
        # return {no_features: self.get_max_auc(no_features, reverse=reverse) }



