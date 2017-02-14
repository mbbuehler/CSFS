from random import shuffle

import numpy as np
from sklearn import tree
from numpy import ravel, interp
from sklearn import datasets, svm, metrics
import pandas as pd
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
# https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
from sklearn.naive_bayes import GaussianNB


class AUCComparator:

    def __init__(self, df, target, fast=True, n_folds=10):
        self.df = df
        self.target = target
        self.fast = fast
        self.classifiers = [
            # GaussianNB(),
            # LogisticRegression(),
            tree.DecisionTreeClassifier(),
            # MLPClassifier(learning_rate_init=1),
            # RandomForestClassifier(),
            # SVC(probability=True), # slows it down a lot
            ]
        self.n_folds = n_folds

    def get_scores(self, feat):
        return [AUCCalculator(self.df, feat, self.target, c, fast=self.fast).calc_mean_auc(self.n_folds) for c in self.classifiers]

    def get_mean_score(self, feat):
        return np.mean(self.get_scores(feat))

class AUCCalculator:
    def __init__(self, df, features, target, classifier, fast=True):
        """

        :param df: pandas.DataFrame
        :param features: list(str)
        :param target: str
        :param classifier: classifier from sklearn
        :return:
        """
        self.df = df
        self.features = features
        self.target = target
        self.classifier = classifier
        self.fast = fast

    def _extract_X_from_df(self):
        return self.df[self.features].values

    def _extract_y_from_df(self):
        return ravel(self.df[self.target].values)

    def calc_mean_auc(self, n_folds):
        if self.fast:
            return self.calc_mean_auc_fast(n_folds)
        return self.calc_mean_auc_slow(n_folds)

    def calc_mean_auc_fast(self, n_folds):
        X = self._extract_X_from_df()
        y = self._extract_y_from_df()
        return np.mean([roc_auc_score(y[test], self.classifier.fit(X[train], y[train]).predict_proba(X[test])[:,1]) for i, (train,test) in enumerate(StratifiedKFold(y, n_folds=n_folds))])

    def calc_mean_auc_slow(self, n_folds):
        X = self._extract_X_from_df()
        y = self._extract_y_from_df()

        cv = StratifiedKFold(y, n_folds=n_folds)
        classifier = self.classifier

        roc_aucs = list()
        for i, (train,test) in enumerate(cv):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1])
            roc_aucs.append(auc(fpr, tpr))

        with_p_r_f = False # with precision, recall, f-score
        if with_p_r_f:
            metadata = dict(Precision=list(), Recall=list(), FScore=list())
            for i, (train,test) in enumerate(cv):
                predicted = classifier.fit(X[train], y[train]).predict(X[test])

                p, r, f, support = precision_recall_fscore_support(y[test], predicted, average='binary')
                metadata['Precision'].append(p)
                metadata['Recall'].append(r)
                metadata['FScore'].append(f)
            for p in sorted(metadata):
                print('{}: {}'.format(p, np.mean(metadata[p])))
        auc_mean = np.mean(roc_aucs)
        return auc_mean

def select_random_features(features, n):
    shuffle(features)
    return features[:n]



def test4():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    df.columns = ['A','B','C','D']
    df['T'] = iris.target
    df = df[df['T']<2] # binary
    AUCComparator(df, ['A','B'], ['B','C'], 'T').compare()

def test3():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    df.columns = ['A','B','C','D']
    df['T'] = iris.target
    df = df[df['T']<2] # binary
    avg_auc = AUCCalculator(df, ['B','C'], 'T', LogisticRegression()).calc_mean_auc(10)
    print(avg_auc)


def test2():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    df.columns = ['A','B','C','D']
    df['T'] = iris.target
    df = df[df['T']<2] # binary
    fs1 = ['A', 'B']
    fs2 = ['B', 'C']
    comparator = AUCComparator(df, fs1, fs2, 'T')

    X = df[fs1].values
    y = ravel(df[['T']].values)

    cv = StratifiedKFold(y, n_folds=10)
    classifier = LogisticRegression()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)
    all_tpr = []
    roc_aucs = list()

    for i, (train,test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, label="ROC fold %d (auc = %0.2f)" % (i, roc_auc))

    plt.plot([0,1], [0,1], '--', color=(.6,.6,.6), label='luck')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    auc_mean = np.mean(roc_aucs)
    print('avg roc auc: {}'.format(auc_mean))
    return auc_mean




def test():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    df.columns = ['A','B','C','D']
    df['T'] = iris.target
    df = df[df['T']<2] # binary
    fs1 = ['A', 'B']
    fs2 = ['B', 'C']
    comparator = AUCComparator(df, fs1, fs2, 'T')

    X_train, X_test, y_train, y_test = train_test_split(df[fs1].values,df[['T']].values, test_size=0.7)

    y_train = ravel(y_train)
    classifier = LogisticRegression()
    classifier = classifier.fit(X_train, y_train)

    y_pred = classifier.predict_proba(X_test) # [[ 0.14388798  0.85611202][... ...]] probabilities of classifier
    y_pred = y_pred[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(roc_auc))
    plt.plot([0,1], [0,1], '--', color=(.6,.6,.6), label='luck')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def measure_time():
    setup = """
from random import shuffle

import numpy as np
from sklearn import tree
from numpy import ravel, interp
from sklearn import datasets, svm, metrics
import pandas as pd
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
# https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from __main__ import AUCComparator
from __main__ import AUCCalculator
features = ['A', 'B', 'C']
data = {'A': [0,1,2,3,4,5], 'B': [0,0,0,1,1,1], 'C' : [1,3,5,2,3,4], 'T': [0,0,0,1,1,1]}
df = pd.DataFrame(data)
    """
    slow = """AUCComparator(df, 'T', n_folds=3, fast=False).get_mean_score(features)"""
    fast = """AUCComparator(df, 'T', n_folds=3, fast=True).get_mean_score(features)"""
    print (min(timeit.Timer(slow, setup=setup).repeat(7, 100)))
    print (min(timeit.Timer(fast, setup=setup).repeat(7, 100)))

if __name__ == '__main__':
    import timeit
    features = ['A', 'B', 'C']
    data = {'A': [0,1,2,3,4,5], 'B': [0,0,0,1,1,1], 'C' : [1,3,5,2,3,4], 'T': [0,0,0,1,1,1]}
    df = pd.DataFrame(data)
    print(AUCComparator(df, 'T', n_folds=3, fast=False).get_mean_score(features))
    print(AUCComparator(df, 'T', n_folds=3, fast=True).get_mean_score(features))

