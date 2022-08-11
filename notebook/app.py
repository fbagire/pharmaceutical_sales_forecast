from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import warnings
import pandas as pd
import pickle
import os

warnings.filterwarnings("ignore")
import mlflow

app = Dash(__name__)
logged_model = 'runs:/08112d9c859c49a994a08ea6c0c80fe2/model'


# new runid "aa2b01ebde4d495088bf8054ef5c8453"
def load_model():
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    client = mlflow.tracking.MlflowClient()

    # local_dir = "/tmp/artifact_downloads"

    # if not os.path.exists(local_dir):
    #     os.mkdir(local_dir)

    # local_path = client.download_artifacts('124bf7c43bdf4733a0a17c5d4435da71', '', local_dir)

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
                html.H3('Sales predictions Scaled on Train Sales (0-1)'),
                dbc.Alert(id='predictions')
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
    sc = pickle.load(open('scaler_new.pkl', 'rb'))
    df_scaled = df.copy()
    loaded_model = load_model()
    y_pred = loaded_model.predict(pd.DataFrame(df_scaled))
    y_pred = sc.inverse_transform(y_pred.reshape(-1, 1))
    df_scaled['predicted_sales'] = y_pred

    return str(y_pred)


if __name__ == '__main__':
    app.run_server(debug=True)
