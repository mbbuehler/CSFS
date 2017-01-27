import plotly.graph_objs as go
import numpy as np

# alphas = [0.3, 0.6, 1]
# sec = 80
# colours = {1:['rgba(255, {}, {}, {})'.format(sec, sec, x) for x in alphas],
#             2: ['rgba( {}, 255,  {}, {})'.format(sec, sec, x) for x in alphas],
#                 3: ['rgba( {},  {}, 255, {})'.format(sec, sec, x) for x in alphas],
#                     4: ['rgba(0, 0, 0, {})'.format(x) for x in alphas],
#                         5: ['rgba(100, 100, 100, {})'.format(x) for x in alphas],
#                             6: ['rgba(200, 150, 0, {})'.format(x) for x in alphas]
#            }
# colours.update({i: ['rgba(200, 150, 0, {})'.format(x) for x in alphas] for i in range(7,20)})


class CIVisualiser:

    def __init__(self, title, x_title, y_title):
        self.title = title
        self.x_title = x_title
        self.y_title = y_title

    def visualise(self, df):
        pass

    @classmethod
    def get_fig(cls, df, col_keys, x_title, y_title, title):
        """

        :param df:
        :param col_keys: list
        :param x_title:
        :return:
        """
        data_nested = [cls.get_traces(df[col_key], col_key) for col_key in col_keys]
        # print(data_nested)
        data = list()
        for group in data_nested:
            for d in group:
                data.append(d)
        # print(data)

        layout = go.Layout(
                title='{}: performance vs. {}'.format(title, x_title),
                xaxis=dict(
                    title=x_title,
                ),
                yaxis=dict(
                    range=[0.5, 1],
                    title=y_title,
                ),
            )

        fig = go.Figure(data=data, layout=layout)
        return fig

    @staticmethod
    def get_traces(df, cond):
        alphas = [0.3, 0.6, 1]
        c = [np.random.randint(0, 256) for i in range(3)]
        colours = ["rgba({},{},{},{})".format(c[0], c[1], c[2], x) for x in alphas]

        trace_auc = go.Scatter(
            x=df.index,
            y=df['mean'],
            name='AUC for {} features'.format(cond),
            line = dict(
                color = colours[0]
            )
            )

        trace_ci_low = go.Scatter(
            x=df.index,
            y=df['ci_lo'],
            fill=None,
            name='{} CI low'.format(cond),
            line = dict(
                color = colours[1]
            )
            )

        trace_ci_hi = go.Scatter(
            x=df.index,
            y=df['ci_hi'],
            fill='tonexty',
            name='{} CI high'.format(cond),
            line = dict(
                color = colours[2]
            )
            )

        return [trace_auc, trace_ci_low, trace_ci_hi]