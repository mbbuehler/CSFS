import re

import numpy as np
import pandas as pd


class ERParser:
    def __init__(self, path_numbered_features):
        self.df_features = pd.read_csv(path_numbered_features)

    def _extract_index_value(self, ranking):
        indeces = list()
        values = list()
        for element in ranking.split(','):
            match = re.search(r'(\d+):(\d+)', element)
            index = match.group(1)
            value = match.group(2)
            indeces.append(index)
            values.append(value)
        # need integers
        return dict(index=np.array(indeces, dtype='int'), No=np.array(values, dtype='int'))

    def parse(self, token):
        ranking = re.search(r'^(.*)\|.+$', token).group(1)
        df_ranking = pd.DataFrame(self._extract_index_value(ranking))
        df_merged = pd.merge(df_ranking, self.df_features, on='No')
        df_merged.index = list(df_merged['index'])
        df_merged = df_merged.drop('index', axis=1)
        print(df_merged)
        return df_merged

    def get_ordered_features(self, token):
        """
        Extracts the ordered list of features from token
        :param token: str e.g. '0:13,1:14,2:1,3:3,4:4,5:7,6:11,7:5,8:12,9:15,10:6,11:2,12:8,13:9,14:10|e7cf0fccca7858d47a96c82837e6d439'
        :return: list(str)
        """
        df_merged = self.parse(token)
        return list(df_merged['Feature'])



def test():
    token = '0:13,1:14,2:1,3:3,4:4,5:7,6:11,7:5,8:12,9:15,10:6,11:2,12:8,13:9,14:10|e7cf0fccca7858d47a96c82837e6d439'
    path = '../datasets/student/evaluation/student_base.csv'
    ERParser(path).parse(token)

    features = ERParser(path).get_ordered_features(token)
    print(features)

if __name__ == '__main__':
    test()