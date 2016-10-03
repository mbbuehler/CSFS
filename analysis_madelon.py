import numpy as np
from sklearn.preprocessing import binarize

from CSFSLoader import CSFSLoader
from CSFSEvaluator import CSFSEvaluator
from noise_helper_funcs import structure_data


def main():
    path = 'datasets/madelon/madelon_combined.csv'
    dataset_name = "madelon"
    target = 'target'
    N_features = 16
    N_samples = 100

    df = CSFSLoader.load_dataset(path, format='csv')
    df = preprocess(df)

    evaluator = CSFSEvaluator(df, target)
    aucs = evaluator.evaluate(N_features, N_samples)
    evaluator.plot(aucs, {'dataset': dataset_name, 'N_features': N_features, 'N_samples': N_samples})

def preprocess(data):
    for f in data:
        b = binarize(data[f], np.mean(data[f]))[0]
        data[f] = b
    # data_structured = structure_data(data[:50])
    return data

if __name__ == "__main__":
    main()