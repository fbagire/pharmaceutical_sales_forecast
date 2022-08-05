import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from myscripts.logger_comb import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


class Modeller:
    """
    - this class is responsible for models preparation and score retrieving
    """

    def __init__(self, df):
        """
        - Initialization of the class
        """
        self.df = df

    def split_data(self, encoded=False):
        """
        - responsible for splitting the data
        """
        train_x = self.df[self.df.columns.difference(['Sales', 'Customers'])]
        train_y = self.df['Sales']

        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, shuffle=False, test_size=0.2,
                                                          random_state=42)
        return X_train, X_val, y_train, y_val

    def error_calculate(self, y_actual, y_predicted):
        """
        - this function finds handle model performance evaluation
        """
        mse = mean_squared_error(y_actual, y_predicted)
        rmse = mean_squared_error(y_actual, y_predicted, squared=False)
        r2 = r2_score(y_actual, y_predicted)

        return mse, rmse, r2

    def feature_importance(self, model, x_train):
        """
        - an algorithm for checking feature importance in case of Random Forests or supported Models
        """
        # initialization
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.show()


if __name__ == "__main__":
    logger.info('script initiated')
