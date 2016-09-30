from CSFSSelector import CSFSRandomSelector, CSFSBestActualSelector, CSFSBestUncertainSelector
from CSFSLoader import CSFSLoader
from CSFSEvaluator import CSFSEvaluator
import pandas as pd
import matplotlib.pyplot as plt

def main2():
    path = 'datasets/test/test.csv'
    target = 'T'
    N_features = 3
    N_samples = 100

    df = CSFSLoader.load_dataset(path, format='csv')
    evaluator = CSFSEvaluator(df, target)
    aucs = evaluator.evaluate(N_features, N_samples)
    evaluator.plot(aucs, {'dataset': 'test', 'N_features': N_features, 'N_samples': N_samples})


def main():
    path = 'datasets/test/test.csv'
    target = 'T'
    N_features = 3
    df = CSFSLoader.load_dataset(path, format='csv')
    print(df)

    selector = CSFSRandomSelector(df, target)
    random_f = selector.select(N_features)

    selector = CSFSBestActualSelector(df, target)
    best_f = selector.select(N_features)

    # df_crowd = pd.DataFrame()
    selector = CSFSBestUncertainSelector(df, target)
    best_noisy_f = selector.select(N_features)

    evaluator = CSFSEvaluator(df, target)
    evaluator.evaluate(random_f)
    evaluator.evaluate(best_f)
    evaluator.evaluate(best_noisy_f)

    randomSelector = CSFSRandomSelector(df, target)
    bestSelector = CSFSBestActualSelector(df, target)
    bestNoisySelector = CSFSBestUncertainSelector(df, target)

    N_samples = 100
    aucs = {'random': [], 'best': [], 'best_noisy': []}
    for i in range(N_samples):
        random_f = randomSelector.select(N_features)
        best_f = bestSelector.select(N_features)
        best_noisy_f = bestNoisySelector.select(N_features)

        aucs['random'].append(evaluator.evaluate(random_f))
        aucs['best'].append(evaluator.evaluate(best_f))
        aucs['best_noisy'].append(evaluator.evaluate(best_noisy_f))

    eval_df = pd.DataFrame(aucs)
    evaluator.plot(eval_df, {'dataset': 'test', 'N_features': N_features, 'N_samples': N_samples})
    # print(eval_df)
    # eval_df.plot()
    # plt.show()




if __name__ == "__main__":
    main2()