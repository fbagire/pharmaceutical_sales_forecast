import warnings
warnings.filterwarnings('ignore')
import dvc.api
import mlflow
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# +
path = 'data/train_model.csv'
repo='https://github.com/fbagire/sales_predict'
# repo = "C:/Users/Faith Bagire/PycharmProjects/pythonProject/sales_predict"
version = '35544764bb947db124da7cd114e4f5daa0fce62b'

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
        data = pd.read_csv(data_url, index_col=[0])
# -

data

# +
# train=pd.read_csv("data/train.csv",low_memory=False,parse_dates=[3],index_col=0)
# print(train.shape)
# train.head()

# +
# test=pd.read_csv("data/test.csv",parse_dates=[4],index_col=0)
# print(test.shape)
# test.head()
# -

# ### Data PreProcessing

# Remove Missing Values
