import plotly.graph_objs as go

fig = go.Figure(
    data=[go.Bar(x=['A', 'B', 'C'], y=[120, 150, 180])],
    layout=go.Layout(
        yaxis=dict(
            tick0=100,
            dtick=20
        )
    )
)

fig.show()
