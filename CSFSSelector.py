from random import shuffle

from infoformulas_listcomp import IG


class CSFSSelector:

    def __init__(self, df, target):
        self.target = target
        self.all_features = [f for f in df]
        self.all_predictors = [f for f in df if f != target]

    def select(self, n):
        raise Exception('Must implement subclass')

    def _check_predictors_length(self, n):
        if len(self.all_predictors) < n:
            raise Exception('n > than len(all_predictors). Not enough features available')


class CSFSRandomSelector(CSFSSelector):

    def select(self, n):
        self._check_predictors_length(n)
        tmp_predictors = self.all_predictors.copy()
        shuffle(tmp_predictors)
        return tmp_predictors[:n]


class CSFSBestActualSelector(CSFSSelector):

    def __init__(self, df, target):
        super().__init__(df, target)
        self.dict_ig = self._get_dict_ig(df, self.target)
        self.ordered_predictors_dsc = self._get_orderend_predictors_dsc(self.dict_ig)

    def select(self, n):
        self._check_predictors_length(n)
        return self.ordered_predictors_dsc[:n]

    def _get_dict_ig(self, df, target):
        return {f: IG(df[target], df[f]) for f in self.all_predictors}

    def _get_orderend_predictors_dsc(self, dict_ig):
        return sorted(dict_ig, key=dict_ig.__getitem__, reverse=True)


