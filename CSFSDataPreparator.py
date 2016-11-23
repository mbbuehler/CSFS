import pandas as pd


class DataPreparator:

    def __init__(self):
        pass

    def drop_columns(self, df, columns):
        return df.drop(columns, axis='columns')

    def _is_binary(self, series):
        return len(set(series)) == 2

    def _is_numerical(self, series):
        dtype = series.dtype
        return dtype == 'float64' or dtype == 'int64'

    def _encode_binary(self, series):
        values = sorted(set(series))
        name = series.name
        result = series.apply(lambda x: 1 if x==values[1] else 0)
        result.name = '{}=={}'.format(name, values[1])
        return result

    def _binning_numerical(self, series, no_bins):
        binned_values = pd.cut(series, no_bins)
        dummies = self._encode_nominal(binned_values)
        return dummies

    def _encode_nominal(self, series):
        name = series.name
        dummies = pd.get_dummies(series, prefix='{}'.format(name))
        return dummies

    def get_encoded(self, series, no_bins):
        """

        :param series:
        :return: pd. Dataframe
        """
        # print(series[:3])

        if self._is_binary(series):
            df_result = self._encode_binary(series)
            # print('is binary')
        elif self._is_numerical(series):
            df_result = self._binning_numerical(series, no_bins=no_bins)
            # print('is numerical')
        else:
            df_result = self._encode_nominal(series)
            # print('is else')
        # print(df_result[:5])
        df_result = df_result.astype('int64')

        # input()
        return df_result

    def prepare(self, df_raw, columns_to_remove=list(), no_bins=3, columns_to_ignore=list()):
        """
        binarises dataframe.
        :return: pd.DataFrame() with binary {0,1} columns
        """
        # print(df_raw)
        df_result = df_raw[columns_to_ignore]
        df_raw = self.drop_columns(df_raw, columns_to_ignore)
        df_raw = self.drop_columns(df_raw, columns_to_remove)
        for f in df_raw:
            df_encoded = self.get_encoded(df_raw[f], no_bins)
            df_result = pd.concat([df_result, df_encoded], axis='columns')
        return df_result



def test():
    data = {'BN_INT': [0, 1, 1, 1, 0, 0, 0, 1], # binary numerical
            'INT': [0, 2, 3, 0, 6, 4, 5, 5], # numerical
            'BN_FLT': [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], # numerical binary float
            'FLT': [0.0, 1.0, 3.0, 4.0, 0.0, 1.0, 1.0, 0.0],
            'BN_NO': ['bstring0', 'bstring1', 'bstring0', 'bstring0', 'bstring1', 'bstring0', 'bstring1', 'bstring0'], # binary nominal
            'NO': ['stringA', 'stringB', 'stringA', 'stringC', 'stringA', 'stringC', 'stringA', 'stringC'], # nominal
            'U': [1, 2, 3, 4, 5, 6, 7, 8] # uselesse (e.g. id)
            }
    df_raw = pd.DataFrame(data)
    columns_to_remove = ['U']
    preparator = DataPreparator(df_raw, columns_to_remove=columns_to_remove)
    df_clean = preparator.prepare()
    assert len(df_clean.columns) == 12

if __name__ == '__main__':
    test()