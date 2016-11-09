import subprocess

import pandas as pd
import numpy as np
import re

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from util.util_features import get_features_from_questions


def get_dataset():
    path_dataset = '../datasets/olympia/cleaned/experiment2/Olympic2016_raw_plus.csv'
    path_questions = '../datasets/olympia/questions/experiment2/featuresOlympia_hi_lo_combined.csv'
    df_raw = pd.read_csv(path_dataset)
    # binarise target variable
    df_raw['medals'] = df_raw['medals'].apply(lambda x: 0 if x == 0 else 1)
    # kick all features we don't want
    features = get_features_from_questions(path_questions)
    df = df_raw[features]
    return df


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.
    http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html accessed: 9.11.16
    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

df = get_dataset()

df = df.dropna(axis='rows')
r = 100

# get a sample
df_sample = df.sample(n=r, axis='rows')

features = list(df_sample.columns)
features.remove('medals')
target = 'medals'

X = df_sample[features]
y = df_sample[target]
dt = DecisionTreeClassifier(criterion='entropy', max_features=3)
dt.fit(X, y)
visualize_tree(dt, features)



