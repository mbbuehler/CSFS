from CSFSLoader import CSFSLoader
from CSFSSelector import CSFSBestActualSelector
from analysis_madelon import preprocess
from infoformulas_listcomp import IG_fast


def test_IG_fast():
    path = 'datasets/madelon/madelon_combined.csv'
    dataset_name = "madelon"
    target = 'target'
    N_features = 3
    N_samples = 100

    df = CSFSLoader.load_dataset(path, format='csv')
    df = preprocess(df)
    selector = CSFSBestActualSelector(df, 'target')
    # igs_fast = IG_fast(df, 'target')
    # print(igs_fast)

    igs = selector._get_dict_ig()
    print(igs)


test_IG_fast()