import plotly.graph_objs as go


alphas = [0.3, 0.6, 1]
sec = 80
colours = {1:['rgba(255, {}, {}, {})'.format(sec, sec, x) for x in alphas],
            2: ['rgba( {}, 255,  {}, {})'.format(sec, sec, x) for x in alphas],
                3: ['rgba( {},  {}, 255, {})'.format(sec, sec, x) for x in alphas],
                    4: ['rgba(0, 0, 0, {})'.format(x) for x in alphas],
                        5: ['rgba(100, 100, 100, {})'.format(x) for x in alphas],
                            6: ['rgba(200, 150, 0, {})'.format(x) for x in alphas]
           }
colours = colours.update({i: ['rgba(200, 150, 0, {})'.format(x) for x in alphas] for i in range(7,20)})

class CIVisualiser:

    def visualise(self, df):
        pass

    @classmethod
    def get_fig(cls, df, first_key, xTitle):
        data = cls.get_traces(df[first_key], first_key, 'no_answers')

        layout = go.Layout(
                title='Performance vs. {}'.format(xTitle),
                xaxis=dict(
                    title=xTitle,

                ),
                yaxis=dict(
                    range=[0.5, 1],
                ),
            )

        fig = go.Figure(data=data, layout=layout)
        return fig

    @staticmethod
    def get_traces(df, cond, name):
        trace_auc = go.Scatter(
            x=df.index,
            y=df['mean'],
            name='{} AUC {}'.format(cond, name),
            line = dict(
                color = colours[cond][0]
            )
            )

        trace_ci_low = go.Scatter(
            x=df.index,
            y=df['ci_lo'],
            fill=None,
            name='{} CI low {}'.format(cond, name),
            line = dict(
                color = colours[cond][1]
            )
            )

        trace_ci_hi = go.Scatter(
            x=df.index,
            y=df['ci_hi'],
            fill='tonexty',
            name='{} CI high {}'.format(cond, name),
            line = dict(
                color = colours[cond][2]
            )
            )

        return [trace_auc, trace_ci_low, trace_ci_hi]