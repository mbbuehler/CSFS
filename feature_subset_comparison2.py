import numpy
from sklearn import tree
from numpy import ravel, interp
from sklearn import datasets, svm, metrics
import pandas as pd
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class AUCComparator:

    def __init__(self, df, features1, features2, target):
        self.df = df
        self.features1 = features1
        self.features2 = features2
        self.target = target
        self.classifiers = [
            GaussianNB(),
            LogisticRegression(),
            tree.DecisionTreeClassifier(),
            RandomForestClassifier(),
            SVC(probability=True),
            ]
        self.n_folds = 2

    def get_scores(self, features):
            return [AUCCalculator(self.df, features, self.target, c).calc_mean_auc(self.n_folds) for c in self.classifiers]

    def compare(self):
        scores_fs1 = self.get_scores(self.features1)
        scores_fs2 = self.get_scores(self.features2)
        print(scores_fs1)
        print(scores_fs2)
        df = pd.DataFrame({'fs1 (mean: %.2f)' % numpy.mean(scores_fs1) : scores_fs1,
                           'fs2 (mean: %.2f)' % numpy.mean(scores_fs2): scores_fs2,
                        }).plot()
        plt.ylabel('AUC')
        plt.xlabel('Classifier')
        plt.xlim([0,1.1])
        plt.ylim([0,1.1])

        plt.show()


class AUCCalculator:
    def __init__(self, df, features, target, classifier):
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

    def _extract_X_from_df(self):
        return self.df[self.features].values

    def _extract_y_from_df(self):
        return ravel(self.df[self.target].values)

    def calc_mean_auc(self, n_folds):
        X = self._extract_X_from_df()
        y = self._extract_y_from_df()

        cv = StratifiedKFold(y, n_folds=n_folds)
        classifier = self.classifier

        roc_aucs = list()
        for i, (train,test) in enumerate(cv):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1])
            roc_aucs.append(auc(fpr, tpr))
        auc_mean = numpy.mean(roc_aucs)
        return auc_mean

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
    mean_fpr = numpy.linspace(0,1,100)
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
    auc_mean = numpy.mean(roc_aucs)
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

def test5():
    data = {'A': [0,1,2,3,4,5], 'B': [0,0,0,1,1,1], 'C' : [1,3,5,2,3,4], 'T': [0,0,0,1,1,1]}
    df = pd.DataFrame(data)
    print(df)
    AUCComparator(df, ['A','B'], ['B','C'], 'T').compare()

if __name__ == '__main__':
    test5()