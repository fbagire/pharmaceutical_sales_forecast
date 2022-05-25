import dvc.api
import mlflow
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'data/train_v2.csv'
repo = "C:/Users/Faith Bagire/PycharmProjects/pythonProject/sales_predict"
version = "\'v2\'"

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
)

mlflow.set_experiment('sales_predict')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)
    with mlflow.start_run(nested=True) as mlrun:
        data = pd.read_csv(data_url)