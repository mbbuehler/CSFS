# coding: utf-8

import pandas as pd
import pickle


class CSFSLoader:

    @staticmethod
    def load_dataset(path, ignore=[], format='csv'):
        if format == 'csv':
            df = pd.read_csv(path)
            return df.drop(ignore, axis=1)
        elif format == "pickle":
            return pickle.load(open(path, 'rb'))
        else:
            print('invalid input format')
            return None