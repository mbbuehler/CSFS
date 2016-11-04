import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup





# df.set_index(['# answerUser unique', 'answer mean', 'answer std', 'answer count'])
from tabulate import tabulate

from infoformulas_listcomp import H_X_Y_from_series, IG_from_series, _H


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

        def get_answers(e):
            r = re.findall(r'::(\d+).\(\d+%\)\)', e)
            answer = float(r[0])/10
            return answer

        def get_answer_user(e):
            return e

        questions = df_answers.question.apply(get_question)
        answers = df_answers.answer.apply(get_answers)
        answer_users = df_answers.answerUser.apply(get_answer_user)

        clean_df = pd.DataFrame({
            'question': questions,
            'answer': answers,
            'answerUser': answer_users,
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
        p = get_val_or_nan(df, "{}_v0".format(f), 'answer mean')
        std_p = get_val_or_nan(df, "{}_v0".format(f), 'answer std')
        n_p = get_val_or_nan(df, "{}_v0".format(f), 'answer count')
        n_p_unique = get_val_or_nan(df, "{}_v0".format(f), 'answerUser nunique')

        p_0 = get_val_or_nan(df, "{}_0_v0".format(f), 'answer mean')
        std_p_0 = get_val_or_nan(df, "{}_0_v0".format(f), 'answer std')
        n_p_0 = get_val_or_nan(df, "{}_0_v0".format(f), 'answer count')
        n_p_0_unique = get_val_or_nan(df, "{}_0_v0".format(f), 'answerUser nunique')

        p_1 = get_val_or_nan(df, "{}_1_v0".format(f), 'answer mean')
        std_p_1 = get_val_or_nan(df, "{}_1_v0".format(f), 'answer std')
        n_p_1 = get_val_or_nan(df, "{}_1_v0".format(f), 'answer count')
        n_p_1_unique = get_val_or_nan(df, "{}_1_v0".format(f), 'answerUser nunique')
        metadata = {'p': p, 'std p': std_p, 'n p': n_p, 'n unique p': n_p_unique,
                   'p|f=0': p_0, 'std p|f=0': std_p_0, 'n p|f=0': n_p_0, 'n unique p|f=0 ': n_p_0_unique,
                   'p|f=1': p_1, 'std p|f=1': std_p_1, 'n p|f=1': n_p_1, 'n unique p|f=1 ': n_p_1_unique,
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

        features = sorted(set(df_clean.feature.apply(lambda x: re.sub(r'_[01]_v\d', '', x))))
        # print(features)
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

    def get_aggregated_df(self):
        df_clean = self.get_clean_df(self.df_answers)
        df_clean = self.questions_to_features(self.df_questions, df_clean)
        df_metadata = self.get_metadata(df_clean)

        df_ig = self.get_ig_df(df_metadata, self.target)
        return df_ig


def test():
    experiment = 'experiment2'
    base_path = 'datasets/olympia/'
    path_answers = '{}results/{}/answers.xlsx'.format(base_path, experiment)
    path_questions = '{}questions/{}/featuresOlympia_hi_lo.csv'.format(base_path, experiment)
    target = 'medals'

    aggregator = CSFSCrowdAggregator(path_questions, path_answers, target)
    df_aggregated = aggregator.get_aggregated_df()

    out_path = '{}results/{}/aggregated.csv'.format(base_path, experiment)
    df_aggregated.to_csv(out_path, index=True)

if __name__ == '__main__':
    test()
