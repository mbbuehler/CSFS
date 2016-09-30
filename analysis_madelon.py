from CSFSLoader import CSFSLoader
from CSFSEvaluator import CSFSEvaluator

def main():
    path = 'datasets/madelon/combined.csv'
    dataset_name = "madelon"
    target = 'target'
    N_features = 3
    N_samples = 5

    df = CSFSLoader.load_dataset(path, format='csv')
    evaluator = CSFSEvaluator(df, target)
    aucs = evaluator.evaluate(N_features, N_samples)
    evaluator.plot(aucs, {'dataset': dataset_name, 'N_features': N_features, 'N_samples': N_samples})



if __name__ == "__main__":
    main()