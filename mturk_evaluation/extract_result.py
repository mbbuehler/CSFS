import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

base_path = '/home/marcello/studies/bachelorarbeit/workspace/github_crowd-sourcing-for-feature-selection/datasets/olympia/cs_experiments/'
in_path = base_path+'data_olympia1.xlsx'
out_path = base_path+'data_olympia1_aggregated.csv'
df_raw = pd.read_excel(in_path)
# print(df_raw[:3])
df_raw = df_raw[df_raw['answerUser']!='A2BGRGVU9HG0C6']


def get_clean_df(df_raw):
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

    questions = df_raw.question.apply(get_question)
    answers = df_raw.answer.apply(get_answers)
    answer_users = df_raw.answerUser.apply(get_answer_user)

    print(answer_users.value_counts())
    clean_df = pd.DataFrame({
        'question': questions,
        'answer': answers,
        'answerUser': answer_users,
        }
    )
    return clean_df


df_clean = get_clean_df(df_raw)

f = {'answer': ['mean', 'std', 'count'], 'answerUser': [pd.Series.nunique]}
# df = df_clean.groupby('question').apply(f)#lambda x: sum(x['answer']))
df = df_clean.groupby('question').agg(f)

# df.columns = ['# answerUser nunique', 'answer mean', 'answer std', 'answer count']
df.columns = [' '.join(col).strip() for col in df.columns.values] # flatten hierarchical column names http://stackoverflow.com/questions/14507794/python-pandas-how-to-flatten-a-hierarchical-index-in-columns

df['crowd_mean_all'] = np.mean(df_clean['answer'])
df['crowd_std_all'] = np.std(df_clean['answer'])
df['true_mean'] = 0.2
df['true_std'] = 0.4

df.to_csv(out_path, sep=',', quotechar='"')
# df.set_index(['# answerUser unique', 'answer mean', 'answer std', 'answer count'])

