from CSFSSelector import CSFSRandomSelector, CSFSBestActualSelector, CSFSBestUncertainSelector
from CSFSLoader import CSFSLoader
from CSFSEvaluator import CSFSEvaluator
import pandas as pd
import matplotlib.pyplot as plt

def main():
    path = 'datasets/test/test.csv'
    target = 'T'
    N_features = 3
    N_samples = 100

    df = CSFSLoader.load_dataset(path, format='csv')
    evaluator = CSFSEvaluator(df, target)
    aucs = evaluator.evaluate(N_features, N_samples)
    evaluator.plot(aucs, {'dataset': 'test', 'N_features': N_features, 'N_samples': N_samples})



if __name__ == "__main__":
    main()