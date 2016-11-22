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

        print(tabulate(df_crowd[:3], headers='keys'))
        # remove metadata we do not need.
        columns_crowd = list(df_crowd.columns) #['p', 'p|f=0', 'p|f=1', 'median', 'median|f=0', 'median|f=1', 'geomean', 'geomean|f=0', 'geomean|f=1', 'IG']
        remove_cols = ['n p', 'n p|f=0', 'n p|f=1', 'std p', 'std p|f=0', 'std p|f=1']
        columns_crowd = [e for e in columns_crowd if e not in remove_cols]
        columns_actual = ['p', 'p|f=0', 'p|f=1', 'IG']
        # if 'IG' not in df_crowd.columns:
        #     columns.remove('IG')
        df_actual = df_actual[columns_actual]
        df_crowd = df_crowd[columns_crowd]

        df_tmp = df_actual.copy()
        df_tmp['median'] = df_actual['p']
        df_tmp['geomean'] = df_actual['p']
        df_tmp['majority_mean'] = df_actual['p']
        df_tmp['median|f=0'] = df_actual['p|f=0']
        df_tmp['geomean|f=0'] = df_actual['p|f=0']
        df_tmp['majority_mean|f=0'] = df_actual['p|f=0']
        df_tmp['median|f=1'] = df_actual['p|f=1']
        df_tmp['geomean|f=1'] = df_actual['p|f=1']
        df_tmp['majority_mean|f=1'] = df_actual['p|f=1']
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

    def questions_to_features(self):
        """
        Returns the cleaned crowd answers with the corresponding features
        :return:
        """
        df_clean = self.get_clean_df(self.df_answers)
        df = self.questions_to_features(self.df_questions, df_clean)
        return df

    def remove_spammers(self, df_clean):
        """
        For each question, removes all answers from users who answered the question more than once
        :param df_clean:
        :return:
        """
        answer_users_count = df_clean.groupby('feature').answerUser.apply(lambda x: x.value_counts()).reset_index()
        spammers = answer_users_count[answer_users_count['answerUser'] > 1]['level_1']
        df_without_spammers = df_clean[~df_clean['answerUser'].isin(spammers)]
        return df_without_spammers

    def raw_to_clean(self, df_answers):
        """
        Extracts questions and answers from raw answer data
        :param df_answers:
        :return:
        """
        def get_triple_question(e):
            soup = BeautifulSoup(e, 'lxml')
            question_dirty = soup.find('i').contents[0]
            question_clean = re.sub('[\n\t]', '', question_dirty)
            result = [question_clean]

            is_triple = e.count('::')==3
            if is_triple:
                index_second_start = e.index('::')+10
                if '100' in e[:index_second_start]: # we have to go one further
                    index_second_start += 2

                assert e[index_second_start]=='I' # makes sure we do not introduce bugs for other questions
                index_second_end = index_second_start + e[index_second_start:].index(':')
                question2 = e[index_second_start:index_second_end]

                index_third_start = index_second_end + e[index_second_end:].index(':')+10
                index_third_end = index_third_start + e[index_third_start:].index(':')
                question3 = e[index_third_start:index_third_end]
                result.append(question2)
                result.append(question3)
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
        df_merged = df_clean.merge(df_questions, left_on='question', right_on='question')
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
    def __init__(self, df_clean, target=""):
        """
        :param df_clean:
        :param target: can be None -> does not calculate IG
        :return:
        """
        self.df_clean = df_clean
        self.target = target

    def _get_feature_metadata(self, f, df):
        # print(tabulate(df, headers='keys'))
        # exit()
        def get_val_or_nan(df, index, column):
            try:
                return df.loc[index][column]
            except:
                # remove 0.4 when we have real data
                return np.nan


        p = get_val_or_nan(df, "{}".format(f), 'answer mean')
        std_p = get_val_or_nan(df, "{}".format(f), 'answer std')
        n_p = get_val_or_nan(df, "{}".format(f), 'answer count')
        median = get_val_or_nan(df, f, 'answer median')
        geomean = get_val_or_nan(df, f, 'answer gmean')
        maj_mean = get_val_or_nan(df, f, 'answer majority_mean')
        # n_p_unique = get_val_or_nan(df, "{}".format(f), 'answerUser nunique')

        p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer mean')
        std_p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer std')
        n_p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer count')
        median_0 = get_val_or_nan(df, "{}_0".format(f), 'answer median')
        geomean_0 = get_val_or_nan(df, "{}_0".format(f), 'answer gmean')
        maj_mean_0 = get_val_or_nan(df, "{}_0".format(f), 'answer majority_mean')

        p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer mean')
        std_p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer std')
        n_p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer count')
        median_1 = get_val_or_nan(df, "{}_1".format(f), 'answer median')
        geomean_1 = get_val_or_nan(df, "{}_1".format(f), 'answer gmean')
        maj_mean_1 = get_val_or_nan(df, "{}_1".format(f), 'answer majority_mean')

        metadata = {'p': p, 'std p': std_p, 'n p': n_p, 'median': median, 'geomean': geomean, 'majority_mean': maj_mean,
                   'p|f=0': p_0, 'std p|f=0': std_p_0, 'n p|f=0': n_p_0, 'median|f=0': median_0, 'geomean|f=0': geomean_0, 'majority_mean|f=0': maj_mean_0,
                   'p|f=1': p_1, 'std p|f=1': std_p_1, 'n p|f=1': n_p_1, 'median|f=1': median_1, 'geomean|f=1': geomean_1,'majority_mean|f=1': maj_mean_1,
                   }
        return metadata

    def get_metadata(self, df_clean):
        """
        Converts dataframe with separate rows for conditional probabilities to aggregated dataframe. adds metadata: mean, std, count and count unique
        :param df_clean:
        :return:
        """
        def majority_mean(x):
            n = 3
            counts = x.value_counts()
            selected = list(counts.head(n).index)
            maj_mean = np.mean(selected)
            return maj_mean

        f = {'answer': ['mean', 'median', gmean, 'std', 'count', majority_mean], 'answerUser': [pd.Series.nunique]}

        df = df_clean.groupby('feature').agg(f)

        df.columns = [' '.join(col).strip() for col in df.columns.values] # flatten hierarchical column names http://stackoverflow.com/questions/14507794/python-pandas-how-to-flatten-a-hierarchical-index-in-columns

        df['crowd_mean_all'] = np.mean(df_clean['answer'])
        df['crowd_std_all'] = np.std(df_clean['answer'])

        features = sorted(set(df_clean.feature.apply(lambda x: re.sub(r'_[01]', '', x))))
        # print(features)
        # exit()
        data = {}
        for f in features:
            data[f] = self._get_feature_metadata(f, df)

        result = pd.DataFrame(data).transpose()
        # print(tabulate(result, headers='keys'))
        return result

    def get_ig_df(self, df, target):
        h_x = _H([df.loc[target]['p'], 1-df.loc[target]['p']])
        df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x)
        df['IG median'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='median')
        df['IG geomean'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='geomean')
        df['IG majority_mean'] = df.apply(IG_from_series, axis='columns', h_x=h_x, identifier='majority_mean')
        return df

    def aggregate(self):
        df_metadata = self.get_metadata(self.df_clean)
        df_result = df_metadata
        if self.target != "":
            df_result = self.get_ig_df(df_metadata, self.target)
        return df_result

def run():
    """
    example call
    :return:
    """
    experiment = 'experiment2'
    base_path = 'datasets/olympia/'
    path_answers = '{}results/{}/answers_combined.xlsx'.format(base_path, experiment)
    path_questions = '{}questions/{}/featuresOlympia_hi_lo_combined.csv'.format(base_path, experiment)
    target = 'medals'

    aggregator = CSFSCrowdAggregator(path_questions, path_answers, target)
    df_aggregated = aggregator.get_aggregated_df()

    out_path = '{}results/{}/aggregated_combined.csv'.format(base_path, experiment)
    df_aggregated.to_csv(out_path, index=True)

    true_path = '{}cleaned/{}/Olympic2016_raw_plus_bin_metadata.csv'.format(base_path, experiment)

    analyse = CSFSCrowdAnalyser()
    df_combined = analyse.get_combined_df(out_path, true_path)
    df_combined.to_csv('{}results/{}/metadata_combined.csv'.format(base_path, experiment), index=True)


def test():
    path_questions = 'datasets/olympia/questions/experiment2/featuresOlympia_hi_lo_combined.csv'
    path_answers = 'datasets/olympia/results/experiment3/answers_raw.xlsx'
    target = 'medals'
    df_clean = CSFSCrowdCleaner(path_questions, path_answers, target).clean()
    df_aggregated = CSFSCrowdAggregator(df_clean, target).aggregate()
    print(tabulate(df_aggregated, headers='keys'))



    #


if __name__ == '__main__':
    # run()
    test()
