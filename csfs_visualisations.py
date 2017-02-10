import plotly.graph_objs as go
import numpy as np
import statsmodels.stats.weightstats as ssw
import scipy.stats as st
from plotly import tools

class COLORS_HEX:
    BLUE = '#1f77b4'
    ORANGE_DARK = '#ff7f0e'
    ORANGE_BRIGHT = '#FFB50E'
    GREEN = '#2ca02c'
    RED = '#D62728'
    VIOLET = '#AC84E2'
    GREY = '#BABABA'
    YELLOW = '#FFF100'

from application.EvaluationRanking import ERCondition
from bootstrap import CSFSBootstrap
colors = {ERCondition.LAYPERSON: COLORS_HEX.VIOLET,
          ERCondition.DOMAIN: COLORS_HEX.GREEN,
          ERCondition.EXPERT: COLORS_HEX.ORANGE_BRIGHT,
          ERCondition.CSFS: COLORS_HEX.BLUE,
          ERCondition.RANDOM: COLORS_HEX.GREY,
          ERCondition.HUMAN: COLORS_HEX.ORANGE_DARK,
          'Laypeople': COLORS_HEX.VIOLET,
          'Domain Experts': COLORS_HEX.GREEN,
          'Data Scientists': COLORS_HEX.ORANGE_BRIGHT,
          'KrowDD': COLORS_HEX.BLUE,
          'Random': COLORS_HEX.GREY,
          'Human': COLORS_HEX.ORANGE_DARK,
          }

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

    def get_traces(self, df):
        return [go.Box(
            y=list(df[no_answers]),
            name=no_answers,
            marker=dict(
                color='rgb(99,99,99, 140)',
            ),
        ) for no_answers in df]

    def get_layout(self):
        return go.Layout(
            title=self.title,
            xaxis=dict(
                title='# Answers Sampled per Feature',
            ),
            yaxis=dict(
                # range=[0, 0.5],
                title='Delta',
            ),
            showlegend=False,
            font=get_font(),
        )

    def get_figure(self, df):
        data = self.get_traces(df)

        layout = self.get_layout()
        fig = go.Figure(data=data, layout=layout)
        return fig


class HumanVsActualBarChart:
    """
    Relative performance
    """

    def get_histogram_trace(self, df, condition, no_answer):
        # print(df.loc[no_answer, condition])
        return go.Histogram(
            # x=df.index,
            x=list(df.loc[no_answer, condition]),
            name="{} {}".format(condition, no_answer),
        )

    def get_trace(self, df, condition_human):
        # print(df[condition_human]) # index: no features, value: list
        y = [np.mean(l) for l in df[condition_human]]
        list_ci = [CSFSBootstrap.get_ci(l) for l in df[condition_human]]
        ci_delta = [ci[1]-ci[0] for ci in list_ci]
        error_y = [d/2 for d in ci_delta]
        # error = [np.std(l) for l in df[condition_human]]
        error = error_y

        return go.Bar(
            x=df.index,
            y=y,
            name=condition_human,
            error_y=dict(
                type='data',
                array=error,
                visible=True
            ),
            marker=dict(
                color=colors[condition_human]
            )
        )

    def get_layout(self):
        return go.Layout(
            # title='Humans vs. Actual',
            xaxis=dict(
                # title='Number of Features',
            ),
            yaxis=dict(
                range=[0, 1],
                # title='Relative Normalized Performance',
            ),
            font=get_font(),
            showlegend=True,
            legend=dict(
                x=0.3,
                y=1,
                orientation='h',
                font=get_font()
            )
        )

    def get_figure(self, df, feature_range):
        """
        :param df:
        :param feature_range: how many features to show (default 1-9)
        :return:
        """
        conditions = sorted(list(df.columns))
        df = df.loc[feature_range[0]:feature_range[-1]]
        data = [self.get_trace(df, condition) for condition in conditions]
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
        fig['layout'].update(height=1800, title='Histograms Human vs Actual')
        return fig


class HumanComparisonBarChart:

    def get_trace(self, df, feature_range, condition, show_legend):
        # print(df[condition_human]) # index: no features, value: list
        df_sel = df[condition].loc[feature_range]
        y = list(df_sel['mean'])
        ci = list((df_sel['ci_hi'] - df_sel['ci_lo']) / 2)

        return go.Bar(
            x=list(feature_range),
            y=y,
            name=ERCondition.get_string_paper(condition),
            error_y=dict(
                type='data',
                array=ci,
                visible=True
            ),
            marker=dict(
                color=colors[condition]
            ),
            showlegend=show_legend
        )

    def get_layout(self):
        return go.Layout(
            xaxis=dict(
                # title='Number of Features',
            ),
            yaxis=dict(
                range=[0.5, 1],
                # title='AUC and Confidence Interval',
            ),
            font=get_font(),
            showlegend=True,
            legend=dict(
                x=0.3,
                y=1,
                orientation='h',
            )
        )

    def get_figure(self, data, feature_range=range(1, 10), conditions=[1, 2, 3]):
        """

        :param data: dict with key in {'income', 'olympia', 'student'} and value pd.DataFrame with multilevel columns conditions -> {ci_hi, ci_lo, count, mean, std} and index: number of features
        :param feature_range: limit of feature range to visualise
        :param conditions: human conditions
        :return:
        """
        datasets = sorted(list(data.keys()))
        dataset_count = len(datasets)
        fig = tools.make_subplots(rows=1, cols=dataset_count, shared_xaxes=True, subplot_titles=[get_dataset_name_paper(name) for name in datasets], ) #  vertical_spacing=0.05
        show_legend = True
        for i in range(dataset_count):
            df_dataset = data[datasets[i]]
            for condition in conditions:
                trace = self.get_trace(df_dataset, feature_range, condition, show_legend)
                fig.append_trace(trace, 1, i+1)
            show_legend = False
            fig['layout']['yaxis'+str(i+1)].update(range=[0.5, 0.9])

        fig['layout'].update(
            height=400,
            font=get_font(),
            showlegend=True,
            legend=dict(
                x=0.25,
                y=1.4,
                orientation='h',
            )
        )

        return fig


    def get_figure_for_dataset(self, df, feature_range=range(1, 10), conditions=[1, 2, 3]):
        data = [self.get_trace(df, feature_range, condition) for condition in conditions]
        layout = self.get_layout()
        fig = go.Figure(data=data, layout=layout)
        return fig


class CSFSVsHumansBarChart:
    def get_trace(self, df, feature_range, condition, show_legend):
        # print(df[condition_human]) # index: no features, value: list
        y = [np.mean(df.loc[no_features, condition]) for no_features in feature_range]
        list_ci = [CSFSBootstrap.get_ci(df.loc[no_features, condition]) for no_features in feature_range]
        ci_delta = [ci[1]-ci[0] for ci in list_ci]
        error_y = [d/2 for d in ci_delta]

        return go.Bar(
            x=list(feature_range),
            y=y,
            name=condition,
            error_y=dict(
                type='data',
                array=error_y,
                visible=True
            ),
            marker=dict(
                color=colors[condition]
            ),
            showlegend = show_legend
        )
    def get_figure(self, data, feature_range=range(1, 10)):
        """

        :param data: dict with key in {'income', 'olympia', 'student'} and value pd.DataFrame with multilevel columns conditions -> {ci_hi, ci_lo, count, mean, std} and index: number of features
        :param feature_range: limit of feature range to visualise
        :param conditions: human conditions
        :return:
        """
        datasets = sorted(list(data.keys()))
        dataset_count = len(datasets)
        fig = tools.make_subplots(rows=1, cols=dataset_count, subplot_titles=[get_dataset_name_paper(name) for name in datasets]) #  vertical_spacing=0.05
        showlegend = True
        for i in range(dataset_count):
            df_dataset = data[datasets[i]]
            print(datasets[i])
            for condition in df_dataset.columns:
                trace = self.get_trace(df_dataset, feature_range, condition, showlegend)
                fig.append_trace(trace, 1, i+1)
            showlegend = False
            fig['layout']['yaxis'+str(i+1)].update(range=[0.5, 0.9])

        fig['layout'].update(
            height=500,
            font=get_font(),
            showlegend=True,
            legend=dict(
                x=0.42,
                y=1.3,
                orientation='h',
            )
        )

        return fig

def get_colours():
    alphas = [0.3, 0.6, 1]
    c = [np.random.randint(0, 256) for i in range(3)]
    colours = ["rgba({},{},{},{})".format(c[0], c[1], c[2], x) for x in alphas]
    return colours

def get_textfont():
    """
    Use in go.Scatter,...
    :return:
    """
    textfont=dict(
        family='sans serif',
        size=18,
        color='#ff7f0e'
    )
    return textfont

def get_font():
    return dict(family='serif', size=24, color='#000')

def get_dataset_name_paper(identifier):
    NAMES = {
        'income': 'Income',
        'olympia': 'Olympics',
        'student': 'Portuguese',
    }
    return NAMES[identifier]