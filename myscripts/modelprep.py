import pandas as pd
import math
import mlflow
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from myscripts.logger_comb import logger
# from scripts.model_serializer import ModelSerializer
# To evaluate end result we have
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score


class Modeller:
    """
    - this class is responsible for modeling
    """

    def __init__(self, df):
        """
        - Initialization of the class
        """
        self.df = df

    def generate_transformation(self, pipeline, type_, value, trim=None, key=None):
        """
        purpose:
            - generates transformations for the data
        input:
            - string,int and df
        returns:
            - transformation
        """
        transformation = None
        if type_ == "numeric":
            transformation = pipeline.fit_transform(self.df.select_dtypes(include=value))
            if trim:
                transformation = pipeline.fit_transform(
                    pd.DataFrame(self.split_data(key, 0.3, trim)).select_dtypes(include=value))
        elif type_ == "categorical":
            transformation = pipeline.fit_transform(self.df.select_dtypes(exclude=value))
            if trim:
                transformation = pipeline.fit_transform(
                    pd.DataFrame(self.split_data(key, 0.3, trim)).select_dtypes(exclude=value))
        return transformation

    def split_data(self, encoded=False):
        """
        - responsible for splitting the data
        """
        train_x = self.df[self.df.columns.difference(['Sales', 'Customers'])]
        train_y = self.df['Sales']

        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def model(self, model, X_test, **kwargs):
        """
        - model the dataset
        """
        X_train, X_val, y_train, y_val = self.split_data()
        # Define Random Forest Model
        model = model(**kwargs)
        # We fit our model with our train data
        model.fit(X_train, y_train)
        # Then predict results from X_test data
        predicted_data = model.predict(X_test)
        # get accuracy score
        return model, predicted_data

    def error_calculate(self, y_actual, y_predicted):
        """
        - this algorithm finds the log loss
        """
        mse = mean_squared_error(y_actual, y_predicted)
        return mse

    def feature_importance(self, model_, column="yes", **kwargs):
        """
        - an algorithm for checking feature importance in case of Random Forests or supported Models
        """
        # initialization
        model = model_(**kwargs)
        X, y = self.get_columns(column, True)
        model.fit(X, y)
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.show()
        return feat_importances


if __name__ == "__main__":
    logger.info('script initiated')
