import pandas as pd
import numpy as np
import re
from scipy.stats.mstats import gmean
from tabulate import tabulate
from bs4 import BeautifulSoup


from infoformulas_listcomp import IG_from_series, _H


class CSFSCrowdAnalyser:

    def get_combined_df(self, path_crowd_aggregated, path_actual_data):
        """
        Combines crowd answers and true data to one dataframe. makes it easier to compare
        :pre data is already written to csv file
        :return:
        """
        # get the data
        df_crowd = pd.read_csv(path_crowd_aggregated, index_col=0)
        df_actual = pd.read_csv(path_actual_data, index_col=0)

        # remove all features we don't have answers for
        features = df_crowd.index.values
        df_actual = df_actual.loc[features]

        # print(tabulate(df_crowd[:3], headers='keys'))
        # remove metadata we do not need.
        columns_crowd = list(df_crowd.columns) #['p', 'p|f=0', 'p|f=1', 'median', 'median|f=0', 'median|f=1', 'geomean', 'geomean|f=0', 'geomean|f=1', 'IG']
        remove_cols = ['n mean', 'n mean|f=0', 'n mean|f=1', 'std mean', 'std mean|f=0', 'std mean|f=1']
        columns_crowd = [e for e in columns_crowd if e not in remove_cols]
        columns_actual = ['mean', 'mean|f=0', 'mean|f=1', 'IG']
        # print(df_actual)
        # exit()

        df_actual = df_actual[columns_actual]
        df_crowd = df_crowd[columns_crowd]

        df_tmp = df_actual.copy()
        df_tmp['median'] = df_actual['mean']
        df_tmp['geomean'] = df_actual['mean']
        df_tmp['majority_mean'] = df_actual['mean']
        df_tmp['median|f=0'] = df_actual['mean|f=0']
        df_tmp['geomean|f=0'] = df_actual['mean|f=0']
        df_tmp['majority_mean|f=0'] = df_actual['mean|f=0']
        df_tmp['median|f=1'] = df_actual['mean|f=1']
        df_tmp['geomean|f=1'] = df_actual['mean|f=1']
        df_tmp['majority_mean|f=1'] = df_actual['mean|f=1']
        df_tmp['IG geomean'] = df_actual['IG']
        df_tmp['IG median'] = df_actual['IG']
        df_tmp['IG majority_mean'] = df_actual['IG']


        # calculate how good/bad the crowd was
        df_diff = abs(df_tmp-df_crowd)

        # combine results in a multiindex
        df_combined = pd.concat(dict(crowd=df_crowd, actual=df_actual, diff=df_diff), axis='columns')
        return df_combined

class CSFSCrowdCleaner:

    def __init__(self, path_questions, path_answers, target):
        self.df_questions = pd.read_csv(path_questions, header=None)
        self.df_questions.columns = ['feature', 'question'] # as column names are missing, we add them here. (internal use only)
        self.df_answers = pd.read_excel(path_answers)
        self.target = target

    def remove_spammers(self, df_clean):
        """
        For each question, removes all answers from users who answered the question more than once
        :param df_clean:
        :return:
        """
        len_original = len(df_clean)
        answer_users_count = df_clean.groupby('feature').answerUser.apply(lambda x: x.value_counts()).reset_index()

        # df_spammed: df with columns: 'feature', 'level_1', 'answerUser', e.g. 'medals', 'ACS3r2S', 10
        df_spammed = answer_users_count[answer_users_count['answerUser']>1].sort_values(by='answerUser', ascending=False)

        df_spammed.columns = ['feature', 'answerUser', 'count']
        df_spammed.to_csv('spammers.csv')
        # print(tabulate(df_spammed))

        # print(len(df_clean))
        for i,row in df_spammed.iterrows():
            drop_indeces = df_clean[(df_clean['answerUser']==row.answerUser) & (df_clean['feature']==row.feature)].index
            df_clean = df_clean.drop(drop_indeces)
        # print(len(df_clean))
        # exit()
        # spammers = answer_users_count[answer_users_count['answerUser'] > 1]['level_1']
        # df_without_spammers = df_clean[~df_clean['answerUser'].isin(spammers)]
        len_no_spam = len(df_clean)
        print('Removed {} spam answers'.format(len_original - len_no_spam))
        return df_clean

    def raw_to_clean(self, df_answers):
        """
        Extracts questions and answers from raw answer data
        :param df_answers:
        :return:
        """
        def get_triple_question(e):
            e = e.replace('\n', '') # remove new lines
            result = list()
            is_triple = e.count('::')==3
            if is_triple: # normal question
                match = re.search(r'<i>(.*)</i>.*%.*?(\w.*\?).*%.*?(\w.*\?)', e.rstrip())
                if match and match.group(3):
                    q1 = match.group(1).strip()
                    q2 = match.group(2).strip()
                    q3 = match.group(3).strip()
                    result += [q1, q2, q3]
                else:
                    print('not found three questions!')
                    print(e)
                    exit()
            else: # is target question (only one)
                match = re.search(r'<i>(.*\?)', e.rstrip())
                q1 = match.group(1).strip()
                result.append(q1)



            #     index_second_start = e.index('::')+10
            #     if '100' in e[:index_second_start]: # we have to go one further
            #         index_second_start += 2
            #
            #     assert e[index_second_start]=='I' # makes sure we do not introduce bugs for other questions
            #     index_second_end = index_second_start + e[index_second_start:].index(':')
            #     question2 = e[index_second_start:index_second_end]
            #
            #
            #
            #     index_third_start = index_second_end + e[index_second_end:].index(':')+10
            #     if ")" in e[index_third_start]:
            #         index_third_start += 1 # second last questions has 100%, too
            #
            #     index_third_end = index_third_start + e[index_third_start:].index(':')
            #     question3 = e[index_third_start:index_third_end]
            #     # if ")" in question3[:5]:
            #     #     print(e)
            #     result.append(question2)
            #     result.append(question3)
            return result

        def get_answers(e):
            r = re.findall(r'::(\d+).\(\d+%\)', e)
            # exit()
            answer = [float(n[0])/10 for n in r]
            return answer

        def get_answer_user(e):
            return e

        # questions = df_answers.question.apply(get_question())
        questions = df_answers.answer.apply(get_triple_question) # yes, we indeed need to search answer column
        answers = df_answers.answer.apply(get_answers)
        answer_users = df_answers.answerUser.apply(get_answer_user)

        final_questions = list()
        final_answers = list()
        final_answer_users = list()

        for i in range(len(questions)):
            final_questions.append(questions[i][0])
            final_answers.append(answers[i][0])
            final_answer_users.append(answer_users[i])
            if len(answers[i]) > 1:
                final_questions.append(questions[i][1])
                final_questions.append(questions[i][2])
                final_answer_users.append(answer_users[i])
                final_answers.append(answers[i][1])
                final_answers.append(answers[i][2])
                final_answer_users.append(answer_users[i])

        clean_df = pd.DataFrame({
            'question': final_questions,
            'answer': final_answers,
            'answerUser': final_answer_users,
            }
        )

        return clean_df

    def questions_to_features(self, df_questions, df_clean):
        """
        Turns question column into features
        :param df_questions:
        :param df_clean:
        :return:
        """
        # print(tabulate(df_clean, headers='keys'))
        # make sure whitespaces do not kick out valid answers
        def clean_string(x):
            x = x.strip('\n\t')
            x = " ".join(x.split())
            # print(x)
            return x
        df_clean['question'] = df_clean['question'].apply(clean_string)
        df_questions['question'] = df_questions['question'].apply(clean_string)

        df_merged = df_clean.merge(df_questions, left_on='question', right_on='question', how='inner')

        df_lost = df_clean.drop(df_merged.index)
        print('Lost {} answers when merging questions and features'.format(len(df_lost)))
        # print('indeces lost: {}'.format(list(df_lost.index)))

        df_merged = df_merged.drop('question', axis='columns')
        return df_merged

    def clean(self):
        """
        Extracts questions and answers from raw result data. Replaces questions with feature names.
        :return: pd.DataFrame
        """
        df_clean = self.raw_to_clean(self.df_answers)
        df_clean = self.questions_to_features(self.df_questions, df_clean)
        df_clean = self.remove_spammers(df_clean)
        return df_clean


class CSFSCrowdAggregator:
    """
    Cleans and aggregates crowd answers
    """
    def __init__(self, df_clean, target="", mode='normal', fake_features={}):
        """
        :param df_clean:
        :param target: can be None -> does not calculate IG
        :param mode: 'extended' outputs various meta info, e.g. geometric mean,...
        :return:
        """
        self.df_clean = df_clean
        self.target = target
        self.mode = mode
        self.fake_features = fake_features

    class Mode:
        NORMAL = 'normal'
        EXTENDED = 'extended'

    def _get_feature_metadata(self, f, df):
        def get_val_or_nan(df, index, column):
            try:
                return df.loc[index][column]
            except:
                # remove 0.4 when we have real data
                return np.nan

        median = get_val_or_nan(df, f, 'answer median')
        median_0 = get_val_or_nan(df, "{}_0".format(f), 'answer median')
        median_1 = get_val_or_nan(df, "{}_1".format(f), 'answer median')
        metadata = {'median': median,
                   'median|f=0': median_0,
                   'median|f=1': median_1,
                   }

        if self.mode==self.Mode.EXTENDED:
            p = get_val_or_nan(df, "{}".format(f), 'answer mean')
            std_p = get_val_or_nan(df, "{}".format(f), 'answer std')
            n_p = get_val_or_nan(df, "{}".format(f), 'answer count')

            geomean = get_val_or_nan(df, f, 'answer gmean')
            maj_mean = get_val_or_nan(df, f, 'answer majority_mean')
            # n_p_unique = get_val_or_nan(df, "{}".format(f), 'answerUser nunique')

            p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer mean')
            std_p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer std')
            n_p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer count')

            geomean_0 = get_val_or_nan(df, "{}_0".format(f), 'answer gmean')
            maj_mean_0 = get_val_or_nan(df, "{}_0".format(f), 'answer majority_mean')

            p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer mean')
            std_p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer std')
            n_p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer count')

            geomean_1 = get_val_or_nan(df, "{}_1".format(f), 'answer gmean')
            maj_mean_1 = get_val_or_nan(df, "{}_1".format(f), 'answer majority_mean')

            metadata = {'mean': p, 'std mean': std_p, 'n mean': n_p, 'median': median, 'geomean': geomean, 'majority_mean': maj_mean,
                   'mean|f=0': p_0, 'std mean|f=0': std_p_0, 'n mean|f=0': n_p_0, 'median|f=0': median_0, 'geomean|f=0': geomean_0, 'majority_mean|f=0': maj_mean_0,
                   'mean|f=1': p_1, 'std mean|f=1': std_p_1, 'n mean|f=1': n_p_1, 'median|f=1': median_1, 'geomean|f=1': geomean_1,'majority_mean|f=1': maj_mean_1,
                   }

        return metadata

    def get_metadata(self, df_clean):
        """
        Converts dataframe with separate rows for conditional probabilities to aggregated dataframe. adds metadata: mean, std, count and count unique
        :param df_clean:
        :return:
        """
        if self.mode==self.Mode.EXTENDED:
            def majority_mean(x):
                n = 3
                counts = x.value_counts()
                selected = list(counts.head(n).index)
                maj_mean = np.mean(selected)
                return maj_mean

            f = {'answer': ['mean', 'median', gmean, 'std', 'count', majority_mean], 'answerUser': [pd.Series.nunique]}
        else:
            f = {'answer': ['median', 'std', 'count']}

        df = df_clean.groupby('feature').agg(f)

        df.columns = [' '.join(col).strip() for col in df.columns.values] # flatten hierarchical column names http://stackoverflow.com/questions/14507794/python-pandas-how-to-flatten-a-hierarchical-index-in-columns

        features = sorted(set(df_clean.feature.apply(lambda x: re.sub(r'_[01]', '', x))))
        data = {}
        for f in features:
            data[f] = self._get_feature_metadata(f, df)

        result = pd.DataFrame(data).transpose()
        return result

    def get_ig_df(self, df, target):
        h_x = _H([df.loc[target]['median'], 1-df.loc[target]['median']])
        df['IG median'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='median')

        if self.mode==self.Mode.EXTENDED:
            h_x = _H([df.loc[target]['mean'], 1-df.loc[target]['mean']])
            df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x)
            h_x = _H([df.loc[target]['geomean'], 1-df.loc[target]['geomean']])
            df['IG geomean'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='geomean')
            h_x = _H([df.loc[target]['majority_mean'], 1-df.loc[target]['majority_mean']])
            df['IG majority_mean'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='majority_mean')
        return df

    def aggregate(self):
        df_tmp = self.df_clean.copy()
        for f in self.fake_features:
            df_tmp = df_tmp.append({'answer': self.fake_features[f], 'answerUser': 'FAKE', 'feature': f}, ignore_index=True)

        df_metadata = self.get_metadata(df_tmp)
        df_result = df_metadata
        if self.target != "":
            df_result = self.get_ig_df(df_metadata, self.target)
        return df_result

class AnswerFilter:
    """
    Filters out answers that do not belong to xls
    """
    def __init__(self, df_raw):
        self.df_raw = df_raw

    def removeWhereContains(self, string):
        """
        Removes all rows that contain the string
        :param string: "credit at a bank"
        :return:
        """
        n_before = len(self.df_raw)
        def contains_string(row):
            return string in row['answer']
        rows_remove = self.df_raw[self.df_raw.apply(contains_string, axis='columns')]
        index_remove = rows_remove.index
        df_filtered = self.df_raw.drop(index_remove)
        print('min index / max removed: {} / {}'.format(min(rows_remove['id']), max(rows_remove['id'])))
        n_after = len(df_filtered)
        print('dropped {} rows'.format(n_before - n_after))
        return df_filtered

class CSFSCrowdAnswergrouper:

    @staticmethod
    def group(df_clean):
        """
        :param df_clean:
             answer      answerUser                  feature
0        0.1  A1KUAL8PN2X8PK  health_(2.333, 3.667]_0
1        0.3   AMA18W8F60Y2J  health_(2.333, 3.667]_0
2        0.6  A1F1OZ54G177D8  health_(2.333, 3.667]_0
3        0.4   ANVAFB99K5RKP  health_(2.333, 3.667]_0
4        0.6  A3D46S3V9SYXTT  health_(2.333, 3.667]_0

:returns
                      p                  p|f=0                 p|f=1
--------------------  -------------------------------------------------------------------------
Fjob_teacher          [0.1,..., 0.1]    [0.6,..., 0.8]         [0.8,... , 0.9]
Medu_(-0.004, 1.333]  [0.1,..., 0.1]    [0.6,..., 0.8]         [0.8,... , 0.9]
...
"""
        data = {}
        for i, row in df_clean.iterrows():
            if row['feature'][-2:] == '_0':
                feature = row['feature'][:-2]
                column = 'p|f=0'
            elif row['feature'][-2:] == '_1':
                feature = row['feature'][:-2]
                column = 'p|f=1'
            else:
                feature = row['feature']
                column = 'p'
            if feature not in data:
                data[feature] = {'p': list(), 'p|f=0': list(), 'p|f=1': list()}
            data[feature][column].append(row['answer'])
        df_grouped = pd.DataFrame(data).transpose()
        return df_grouped


def run():
    """
    example call
    :return:
    """
    experiment = 'experiment1'
    base_path = 'datasets/income/'
    path_answers = '{}results/{}/answers_raw2.xlsx'.format(base_path, experiment)
    df_raw = pd.read_excel(path_answers)
    df_filtered = AnswerFilter(df_raw).removeWhereContains("credit at a bank")



def test():
    path_questions = 'datasets/olympia/questions/experiment2/featuresOlympia_hi_lo_combined.csv'
    path_answers = 'datasets/olympia/results/experiment3/answers_raw.xlsx'
    target = 'medals'
    df_clean = CSFSCrowdCleaner(path_questions, path_answers, target).clean()
    df_aggregated = CSFSCrowdAggregator(df_clean, target).aggregate()
    print(tabulate(df_aggregated, headers='keys'))



    #


if __name__ == '__main__':
    run()
    # test()
