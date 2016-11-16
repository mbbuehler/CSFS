import pandas as pd
import re

from CSFSSelector import CSFSBestActualSelector



def get_target_ratio(path):
    df = pd.read_csv(path)
    print(df['T'].describe())


def create_dn_names_from_base(dataset_base):
    """
    Creates new dataset names from a prefix (base). e.g. artifical10 becomes artifical10_true, artificial10_noisy
    :param dataset_base: list
    :return: list
    """
    dataset_names = [dn+'_true' for dn in dataset_base] + [dn+'_noisy' for dn in dataset_base]
    return dataset_names


def get_ranked_features():
    base_path = '/home/marcello/studies/bachelorarbeit/workspace/github_crowd-sourcing-for-feature-selection/datasets/olympia/raw/olympic2016_raw_plus/'
    path = base_path+'Olympic2016_raw_plus_bin.csv'

    df = pd.read_csv(path)
    target = 'medals'
    best_selector = CSFSBestActualSelector(df, target)

    dict_ig = best_selector._get_dict_ig()
    ordered_features = sorted(dict_ig, key=dict_ig.__getitem__)
    print('=== All Features')
    print([f for f in df])
    print('=== All Features with IG')
    for f in ordered_features:
        print(f, dict_ig[f])



def get_features_from_questions(path_questions, remove_cond=False, remove_binning=False):
    """
    Extracts all feature names from questions file and returns list of features as string. Does remove target
    :param: remove_cond: removes _1 and _0 at the end of features
    :param remove_binning: removes ond and binning (_[..]_3) at the end of features
    :return: list(str)
    """
    df_features = pd.read_csv(path_questions, header=None)
    features = list(df_features[0])
    # print(features)
    if remove_cond:
        features = remove_cond_markup(features)
    if remove_binning:
        features = remove_binning_cond_markup(features)
    return features

def remove_cond_markup(features):
    """
    Removes _0 or _1 markup from each feature. ALSO REMOVES medals!
    :param features:
    :return:
    """
    result = list()
    for f in features:
        match = re.match(r'(.*[(\[].*[)\]])', f) or re.match(r'(.*_\d)_\d', f)
        if match:
            result.append(match.group(1))
    result = list(set(result))
    return result

def remove_binning_cond_markup(features):
    """
    turns df with index "education expenditures_(4.133, 5.6]_0" into list with "education expenditures"
    :param features: list(str)
    :return: list(str)
    """
    features = list(set([re.sub(r'[(_\d)(\[\()].*$', '', x) for x in features]))
    return features

def get_feature_inf():
    df = pd.read_csv('datasets/olympia/cleaned/experiment2/Olympic2016_raw_plus_bin.csv')
    features = list(df.columns)
    # print(features)
    features_base = remove_binning_cond_markup(features)
    def extract_bins(f):
        match = re.search(r'(-?\d+\.?\d*), (-?\d+\.?\d*)', f)
        if match:
            thresholds = match.group().split(', ')
            return {f: [float(t) for t in thresholds]}
        return {}

    result = dict()
    for f in features:
        result.update(extract_bins(f))

    structured = dict()
    for f_base in features_base:
        structured[f_base] = {}
        for f in result:
            if f.startswith(f_base+'_'):
                structured[f_base][f] = result[f]
    print(structured)
    return structured



if __name__ == '__main__':
    # path = '../datasets/artificial/artificial12.csv'
    # print(get_target_ratio(path))
    # path = '../datasets/olympia/raw/Opympic_0_extra_2016.csv'
    # binarise_dataset()
    # get_ranked_features()

    # base_path = '/home/marcello/studies/bachelorarbeit/workspace/github_crowd-sourcing-for-feature-selection/datasets/olympia/raw/olympic2016_raw_plus/'
    # path = base_path+'Olympic2016_raw_plus.csv'
    #
    # df = pd.read_csv(path)
    # target = 'medals'
    # df['medals'] = df['medals']>0
    # df['medals'] = df['medals'].astype(int)
    # df2 = df[df['electricity consumption per capita']<5612.31]
    # print(df2[target].describe())
    features = ['education expenditures_(4.133, 5.6]',
                'inflation rate_(1.9, 4.2]',
                'region_3',
                'unemployment rate_[0.3, 6.433]',
                'public debt_(61.733, 226.1]',
                'electricity consumption_[90210000, 7952000000]',
                'exports_[1163000, 10071333333.333]',
                'electricity consumption_(55576666666.667, 3890000000000]',
                'internet users_(6149000, 245000000]',
                'exports_(77193333333.333, 1580000000000]',
                ]
    # create_question_templates(10)
    get_feature_inf()