from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import warnings
from sklearn.preprocessing import MinMaxScaler
from preprocess_data import clean_data
from flask_caching import Cache

warnings.filterwarnings("ignore")

import pandas as pd

app = Dash(__name__)


# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory'
# })
# TIMEOUT = 60


# @cache.memoize(timeout=TIMEOUT)
def load_model():
    import mlflow
    logged_model = 'runs:/08112d9c859c49a994a08ea6c0c80fe2/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


# server = app.server
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
                        # [{'id': 'Model', 'name': ''}] +
                        [{'id': p, 'name': p} for p in params]
                    ),
                    data=[
                        dict(Model=i, **{param: 0 for param in params})
                        for i in range(1, 5)
                    ],
                    editable=True
                )
            ]),
        dash_table.DataTable(id='dataframe_in'),
        dbc.Row(
            [
                html.H3('Print Guidance text and then predictions'),
                dbc.Alert(id='predictions'),
                # dash_table.DataTable(id='predictions')

            ])

    ])


@app.callback(
    Output('dataframe_in', 'data'),
    Input('sales_dataframe', 'data'),
    Input('sales_dataframe', 'columns'))
def display_output(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])

    return df.to_dict(orient='records')


@app.callback(
    Output('predictions', 'children'),
    Input('sales_dataframe', 'data'),
    Input('sales_dataframe', 'columns'))
def model_accuracy(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    df = df.astype('float64')
    df_scaled = df.copy()
    loaded_model = load_model()
    y_pred = loaded_model.predict(pd.DataFrame(df_scaled))
    df_scaled['predicted_sales'] = y_pred

    # return df_scaled.to_dict(orient='records')
    return str(y_pred[0])


if __name__ == '__main__':
    app.run_server(debug=True)
