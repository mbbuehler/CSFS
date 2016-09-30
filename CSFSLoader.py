# coding: utf-8

import pandas as pd


class CSFSLoader:

    @staticmethod
    def load_dataset(path, format='csv'):
        if format == 'csv':
            return pd.read_csv(path)
        else:
            print('invalid input format')
            return None