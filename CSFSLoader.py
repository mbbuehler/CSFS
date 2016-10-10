# coding: utf-8

import pandas as pd
import pickle


class CSFSLoader:

    @staticmethod
    def load_dataset(path, format='csv'):
        if format == 'csv':
            return pd.read_csv(path)
        elif format == "pickle":
            return pickle.load(open(path, 'rb'))
        else:
            print('invalid input format')
            return None