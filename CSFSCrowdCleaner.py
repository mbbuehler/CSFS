import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup





# df.set_index(['# answerUser unique', 'answer mean', 'answer std', 'answer count'])

class CSFSCrowdCleaner:
    def __init__(self, path_questions, path_answers):
        self.df_questions = pd.read_csv(path_questions)
        self.df_answers = pd.read_excel(path_answers)

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

    def get_metadata(self, df_clean):
        f = {'answer': ['mean', 'std', 'count'], 'answerUser': [pd.Series.nunique]}
        # df = df_clean.groupby('question').apply(f)#lambda x: sum(x['answer']))
        df = df_clean.groupby('question').agg(f)

        # df.columns = ['# answerUser nunique', 'answer mean', 'answer std', 'answer count']
        df.columns = [' '.join(col).strip() for col in df.columns.values] # flatten hierarchical column names http://stackoverflow.com/questions/14507794/python-pandas-how-to-flatten-a-hierarchical-index-in-columns

        df['crowd_mean_all'] = np.mean(df_clean['answer'])
        df['crowd_std_all'] = np.std(df_clean['answer'])


        # df.to_csv(out_path, sep=',', quotechar='"')

    def work(self):
        df_clean = self.get_clean_df(self.df_answers)
        df_metadata = self.get_metadata(df_clean)
        print(df_metadata)


def test():
    base_path = '/home/marcello/studies/bachelorarbeit/workspace/github_crowd-sourcing-for-feature-selection/datasets/olympia/cs_experiments/results/'
    path_answers = base_path+'data_olympia1.xlsx'
    path_questions = base_path+'Olympic2016_raw_plus_bin_questions_4features_v1.csv'
    cleaner = CSFSCrowdCleaner(path_questions, path_answers)
    cleaner.work()

if __name__ == '__main__':
    test()
