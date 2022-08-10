from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import mlflow
import pandas as pd

app = Dash(__name__)
server = app.server
params = ['Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
          'CompetitionOpenSinceYear', 'DayOfWeek', 'DayOfYear', 'Open', 'Promo',
          'Promo2', 'Promo2SinceWeek', 'PromoInterval', 'SchoolHoliday',
          'StateHoliday', 'Store', 'StoreType', 'Years', 'dayType']

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2('Pharmaceutical Sales Prediction',
                        style={'textAlign': 'center', 'font_family': "Times New Roman", 'color': '#0F562F'}),
            ]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3('Please input values for your stores to get the estimated sales predictions'),
                        'You can read more about features explanation ',
                        dcc.Link("here", href='https://www.kaggle.com/competitions/rossmann-store-sales/data')
                    ]
                )
            ]),
        dbc.Row(
            [
                dash_table.DataTable(
                    id='sales_dataframe',
                    columns=(
                            [{'id': 'Model', 'name': 'Model'}] +
                            [{'id': p, 'name': p} for p in params]
                    ),
                    data=[
                        dict(Model=i, **{param: 0 for param in params})
                        for i in range(1, 5)
                    ],
                    editable=True
                )
            ])
    ])


@app.callback(
    Output('sales_dataframe', 'DataFrame'),
    Input('sales_dataframe', 'data'),
    Input('sales_dataframe', 'columns'))
def display_output(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])

    return df




@app.callback(
    Output('sales_dataframe', 'DataFrame'),
    Input('sales_dataframe', 'data'),
    Input('sales_dataframe', 'columns'))
def display_output(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])

    return df
    # return {
    #     'data': [{
    #         'type': 'parcoords',
    #         'dimensions': [{
    #             'label': col['name'],
    #             'values': df[col['id']]
    #         } for col in columns]
    #     }]
    # }


logged_model = 'runs:/4ae62e352b724d6688d161de367a9961/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model.predict(pd.DataFrame(df))

if __name__ == '__main__':
    app.run_server(debug=True)
