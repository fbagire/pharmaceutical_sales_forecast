import warnings
warnings.filterwarnings('ignore')
import dvc.api
import mlflow
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data_dvc(path, git_revision):

    repo = 'https://github.com/fbagire/sales_predict'
    version = git_revision

    data_url = dvc.api.get_url(path=path, repo=repo, rev=version)

    mlflow.set_experiment('sales_predict')

    if __name__ == '__main__':
        warnings.filterwarnings('ignore')
        np.random.seed(40)
        with mlflow.start_run(nested=True) as mlrun:
            data = pd.read_csv(data_url, index_col=[0])

    return data


train_df=load_data_dvc('data/train_model.csv','70a72e7e4cda6da4ab57bb8571e5cc2f3c5366e0')

test_df=load_data_dvc('data/test_model.csv','70a72e7e4cda6da4ab57bb8571e5cc2f3c5366e0')

train_df.shape,test_df.shape



# ### Data PreProcessing

# Remove Missing Values
