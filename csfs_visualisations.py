import plotly.graph_objs as go
import numpy as np
import statsmodels.stats.weightstats as ssw
import scipy.stats as st
from plotly import tools

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


class AnswerDeltaVisualiserBar:

    def __init__(self, title):
        self.title = title

    def get_trace(self, df, condition):
        return go.Bar(
            x=df.index,
            y=[np.mean(y) for y in df[condition]],
            text=["min: {:.2f} \nmax: {:.2f} \nstd: {:.4f}".format(np.min(y), np.max(y), np.std(y)) for y in df[condition]],
            name=condition
        )

    def get_layout(self):
        return go.Layout(
            title=self.title,
            xaxis=dict(
                title='number of answers sampled per feature (without replacement)',
            ),
            yaxis=dict(
                range=[0, 0.5],
                title='delta (mean difference over all features)',
            ),
            barmode='group'
        )

    def get_figure(self, df):
        def f(row):
            row['diff IG range'] = [abs(np.min(row['IG'])-np.max(row['IG']))]
            row['IG std'] = [abs(np.std(row['IG']))]
            row['p all'] = row['p'] + row['p|f=0'] + row['p|f=1']
            row['median all'] = [np.median(row['p all'])]
            return row
        df = df.apply(f, axis='columns')
        conditions = df.columns

        data = [self.get_trace(df, condition) for condition in conditions]

        layout = self.get_layout()
        fig = go.Figure(data=data, layout=layout)
        return fig

class AnswerDeltaVisualiserLinePlot:

    def __init__(self, title):
        self.title = title

    def get_traces(self, df, condition):
        colours = get_colours()

        my = go.Scatter(
            x=df.index,
            y=[np.mean(y) for y in df[condition]],
            name="{} mean".format(condition),
            line = dict(
                color=colours[0]
            )
        )
        lo = go.Scatter(
            x=df.index,
            y=[ssw.DescrStatsW(y).tconfint_mean()[0] for y in df[condition]],
            name="{} CI low".format(condition),
            fill=None,
            line = dict(
                color = colours[0]
            )
        )
        hi = go.Scatter(
            x=df.index,
            y=[ssw.DescrStatsW(y).tconfint_mean()[1] for y in df[condition]],
            name="{} CI high".format(condition),
            fill='tonexty',
            line = dict(
                color = colours[0]
            )
        )
        return [my, lo, hi]

    def get_layout(self):
        return go.Layout(
            title=self.title,
            xaxis=dict(
                title='number of answers sampled per feature (without replacement)',
            ),
            yaxis=dict(
                range=[0, 0.5],
                title='delta (mean difference over all features)',
            ),
            barmode='group'
        )

    def get_figure(self, df):
        def f(row):
            # row['diff IG range'] = [abs(np.min(row['IG'])-np.max(row['IG']))]
            row['p all'] = row['p'] + row['p|f=0'] + row['p|f=1']
            row['median all'] = [abs(np.median(row['p all']))]
            return row
        df = df.apply(f, axis='columns')
        conditions = df.columns

        data = list()
        for condition in conditions:
            data += self.get_traces(df, condition)

        layout = self.get_layout()
        fig = go.Figure(data=data, layout=layout)
        return fig


class AnswerDeltaVisualiserBox:

    def __init__(self, title):
        self.title = title

    def get_traces(self, df, condition):
        return [go.Box(
            y=list(df.loc[no_answers, condition]),
            name=no_answers,
        ) for no_answers in df[condition].index]

    def get_layout(self):
        return go.Layout(
            title=self.title,
            xaxis=dict(
                title='# Answers Sampled per Feature (without Replacement)',
            ),
            yaxis=dict(
                range=[0, 0.5],
                title='Delta',
            ),
        )

    def get_figure(self, df):
        def f(row):
            row['diff IG range'] = [abs(np.min(row['IG'])-np.max(row['IG']))]
            row['IG std'] = [abs(np.std(row['IG']))]
            row['p all'] = row['p'] + row['p|f=0'] + row['p|f=1']
            row['median all'] = [np.median(row['p all'])]
            return row
        df = df.apply(f, axis='columns')

        data = self.get_traces(df, 'p all')

        layout = self.get_layout()
        fig = go.Figure(data=data, layout=layout)
        return fig


class HumanVsActualBarChart:

    def get_histogram_trace(self, df, condition, no_answer):
        # print(df.loc[no_answer, condition])
        return go.Histogram(
            # x=df.index,
            x=list(df.loc[no_answer, condition]),
            name="{} {}".format(condition, no_answer),
        )


    def get_trace(self, df, condition_human):
        return go.Histogram(
            # x=df.index,
            x=list(df[condition_human]),
            name=condition_human,
        )

    def get_layout(self):
        return go.Layout(
            # title='Humans vs. Actual',
            # xaxis=dict(
            #     title='Number of Answers',
            # ),
            # yaxis=dict(
            #     range=[0, 1],
            #     title='Relative Normalized Performance',
            # ),
        )

    def get_figure(self, df):

        data = [self.get_trace(df, condition) for condition in df]
        layout = self.get_layout()
        fig = go.Figure(data=data, layout=layout)
        return fig

    def get_histograms(self, df):
        conditions = list(df.columns)
        answer_range = list(df.index)
        fig = tools.make_subplots(rows=len(answer_range), cols=len(conditions))
        for i in range(len(answer_range)):
            for j in range(len(conditions)):
                trace = self.get_histogram_trace(df, conditions[j], answer_range[i])
                fig.append_trace(trace, i+1, j+1)
        fig['layout'].update(height=1200, title='Histograms Human vs Actual')
        return fig


def get_colours():
    alphas = [0.3, 0.6, 1]
    c = [np.random.randint(0, 256) for i in range(3)]
    colours = ["rgba({},{},{},{})".format(c[0], c[1], c[2], x) for x in alphas]
    return colours