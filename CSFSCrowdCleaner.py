import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf




# df.set_index(['# answerUser unique', 'answer mean', 'answer std', 'answer count'])
from tabulate import tabulate

from infoformulas_listcomp import H_X_Y_from_series, IG_from_series, _H


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

        # remove metadata we do not need.
        columns = ['p', 'p|f=0', 'p|f=1', 'IG']
        df_actual = df_actual[columns]
        df_crowd = df_crowd[columns]

        # calculate how good/bad the crowd was
        df_diff = abs(df_actual-df_crowd)

        # combine results in a multiindex
        df_combined = pd.concat(dict(crowd=df_crowd, actual=df_actual, diff=df_diff), axis='columns')
        return df_combined




class CSFSCrowdAggregator:
    """
    Cleans and aggregates crowd answers
    """
    def __init__(self, path_questions, path_answers, target):
        self.df_questions = pd.read_csv(path_questions, header=None)
        self.df_questions.columns = ['feature', 'question']
        self.df_answers = pd.read_excel(path_answers)
        self.target = target

    def get_clean_df(self, df_answers):
        def get_question(e):
            soup = BeautifulSoup(e, 'lxml')
            question_dirty = soup.find('i').contents[0]
            question_clean = re.sub('[\n\t]', '', question_dirty)
            return question_clean

        def get_triple_question(e):
            soup = BeautifulSoup(e, 'lxml')
            question_dirty = soup.find('i').contents[0]
            question_clean = re.sub('[\n\t]', '', question_dirty)
            result = [question_clean]

            is_triple = e.count('%')==3
            if is_triple:
                index_second_start = e.index('%')+3
                assert e[index_second_start]=='I' # makes sure we do not introduce bugs for other questions
                index_second_end = index_second_start + e[index_second_start:].index(':')
                question2 = e[index_second_start:index_second_end]

                index_third_start = index_second_end + e[index_second_end:].index(':')+10
                index_third_end =  index_third_start + e[index_third_start:].index(':')
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
        # print(questions)
        answers = df_answers.answer.apply(get_answers)
        # print(answers)
        answer_users = df_answers.answerUser.apply(get_answer_user)

        final_questions = list()
        final_answers = list()
        final_answer_users = list()

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

        # print(final_questions)
        # print(final_answers)
        # exit()
        answer_users = df_answers.answerUser.apply(get_answer_user)

        clean_df = pd.DataFrame({
            'question': final_questions,
            'answer': final_answers,
            'answerUser': final_answer_users,
            }
        )

        return clean_df

    def _get_feature_metadata(self, f, df):
        def get_val_or_nan(df, index, column):
            try:
                return df.loc[index][column]
            except:
                # remove 0.4 when we have real data
                return np.nan


        p = get_val_or_nan(df, "{}".format(f), 'answer mean')
        std_p = get_val_or_nan(df, "{}".format(f), 'answer std')
        n_p = get_val_or_nan(df, "{}".format(f), 'answer count')
        # n_p_unique = get_val_or_nan(df, "{}".format(f), 'answerUser nunique')

        p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer mean')
        std_p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer std')
        n_p_0 = get_val_or_nan(df, "{}_0".format(f), 'answer count')
        # n_p_0_unique = get_val_or_nan(df, "{}_0".format(f), 'answerUser nunique')

        p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer mean')
        std_p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer std')
        n_p_1 = get_val_or_nan(df, "{}_1".format(f), 'answer count')
        # n_p_1_unique = get_val_or_nan(df, "{}_1".format(f), 'answerUser nunique')
        metadata = {'p': p, 'std p': std_p, 'n p': n_p, #'n unique p': n_p_unique,
                   'p|f=0': p_0, 'std p|f=0': std_p_0, 'n p|f=0': n_p_0, #'n unique p|f=0 ': n_p_0_unique,
                   'p|f=1': p_1, 'std p|f=1': std_p_1, 'n p|f=1': n_p_1, #'n unique p|f=1 ': n_p_1_unique,
                   }
        return metadata

    def get_metadata(self, df_clean):
        """
        Converts dataframe with separate rows for conditional probabilities to aggregated dataframe. adds metadata: mean, std, count and count unique
        :param df_clean:
        :return:
        """
        f = {'answer': ['mean', 'std', 'count'], 'answerUser': [pd.Series.nunique]}
        # df = df_clean.groupby('question').apply(f)#lambda x: sum(x['answer']))
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

    def questions_to_features(self, df_questions, df_clean):
        df_merged = df_clean.merge(df_questions, left_on='question', right_on='question')
        df_merged = df_merged.drop('question', axis='columns')
        return df_merged

    def get_ig_df(self, df, target):
        h_x = _H([df.loc[target]['p'], 1-df.loc[target]['p']])
        df['IG'] = df.apply(IG_from_series, axis='columns', h_x=h_x)
        return df

    def get_without_spammers(self, df_clean):
        answer_users_count = df_clean.groupby('feature').answerUser.apply(lambda x: x.value_counts()).reset_index()
        spammers = answer_users_count[answer_users_count['answerUser']>1]['level_1']
        df_without_spammers = df_clean[~df_clean['answerUser'].isin(spammers)]
        return df_without_spammers

    def get_aggregated_df(self):
        df_clean = self.get_clean_df(self.df_answers)

        df_clean = self.questions_to_features(self.df_questions, df_clean)

        df_clean = self.get_without_spammers(df_clean)

        df_metadata = self.get_metadata(df_clean)
        # print(tabulate(df_metadata))
        # exit()

        df_ig = self.get_ig_df(df_metadata, self.target)
        return df_ig

    def run(self):
        """
        example call
        :return:
        """
        experiment = 'experiment2'
        base_path = 'datasets/olympia/'
        path_answers = '{}results/{}/answers.xlsx'.format(base_path, experiment)
        path_questions = '{}questions/{}/featuresOlympia_hi_lo.csv'.format(base_path, experiment)
        target = 'medals'

        aggregator = CSFSCrowdAggregator(path_questions, path_answers, target)
        df_aggregated = aggregator.get_aggregated_df()

        out_path = '{}results/{}/aggregated.csv'.format(base_path, experiment)
        df_aggregated.to_csv(out_path, index=True)

def test():
    experiment = 'experiment2'
    base_path = 'datasets/olympia/'
    path_answers = '{}results/{}/answers.xlsx'.format(base_path, experiment)
    path_questions = '{}questions/{}/featuresOlympia_hi_lo.csv'.format(base_path, experiment)
    target = 'medals'

    out_path = '{}results/{}/aggregated.csv'.format(base_path, experiment)
    true_path = '{}cleaned/{}/Olympic2016_raw_plus_bin_metadata.csv'.format(base_path, experiment)

    analyse = CSFSCrowdAnalyser()
    df_combined = analyse.get_combined_df(out_path, true_path)
    df_combined.iplot()
    # df_combined.to_csv('{}results/{}/combined.csv'.format(base_path, experiment), index=True)


if __name__ == '__main__':
    test()
