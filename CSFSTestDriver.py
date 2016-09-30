from CSFSSelector import CSFSRandomSelector, CSFSBestActualSelector
from CSFSLoader import CSFSLoader



def main():
    path = 'datasets/test/test.csv'
    target = 'T'
    df = CSFSLoader.load_dataset(path, format='csv')
    print(df)

    selector = CSFSRandomSelector(df, target)
    for i in range(3):
        pred_random = selector.select(2)
        print(pred_random)
        assert len(pred_random) == 2

    selector = CSFSBestActualSelector(df, target)
    pred_best = selector.select(3)
    print(pred_best)



if __name__ == "__main__":
    main()