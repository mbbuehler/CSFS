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

def kick_non_2016(path):
    df_all = pd.read_csv(path)
    df_2016 = df_all[df_all['year']==2016]
    print(len(df_all))
    print(df_all[:10])
    all_countries = set(df_all.country.values)
    print(len(set(all_countries)))
    # print(df_2016)
    countries = set(df_2016.country.values)
    print([c for c in all_countries if c not in countries])# ['Serbia and Montenegro'] is missing
    df_2016.to_csv('../datasets/olympia/raw/Olympic2016_raw.csv', index=False)

def binarise_dataset():
    base_path = '/home/marcello/studies/bachelorarbeit/workspace/github_crowd-sourcing-for-feature-selection/datasets/olympia/raw/olympic2016_raw_plus/'
    path = base_path+'Olympic2016_raw_plus.csv'
    df = pd.read_csv(path)
    df = df.drop('country', axis=1)
    df = df.dropna(axis='index')
    df['medals'] = df['medals']>0
    df['medals'] = df['medals'].astype(int)

    binary_features = ['medals', 'host', 'antehost', 'planned_econ', 'rel_muslim']
    features_to_bin = ['continent', 'region']
    features_to_do = [f for f in df if f not in binary_features and f not in features_to_bin]
    # binarise
    for f in features_to_bin:
        df = df.combine_first(pd.get_dummies(df[f], prefix=f))
        df = df.drop(f, axis='columns')
    # bin an binarise

    for f in features_to_do:
        print(f)
        bins = pd.qcut(df[f], 3)
        for b in sorted(set(bins)):
            df['{}_{}'.format(f, b)] = 0
        for i,b in enumerate(bins):
            df['{}_{}'.format(f, b)].iloc[i] = 1
        df = df.drop(f, axis='columns')


    df.to_csv(base_path+'Olympic2016_raw_plus_bin.csv', index=False)


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

def get_features_from_questions(path_questions, remove_cond=False, remove_binning=False):
    """
    Extracts all feature names from questions file and returns list of features as string. Does not remove target
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
        print(f)
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