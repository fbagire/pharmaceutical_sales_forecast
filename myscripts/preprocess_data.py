import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class clean_data():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # fixing outliers
    def fix_outliers(self, df: pd.DataFrame, column) -> pd.DataFrame:

        """" Replace the outliers with mean value This helps with the computational analysis.
        """
        bounds = {}
        for col in column:
            q1 = df[col].quantile(0.05)  # 0.05
            q3 = df[col].quantile(0.95)  # 0.95
            lower_b = q1 - (1.5 * (q3 - q1))
            upper_b = q3 + (1.5 * (q3 - q1))
            bounds[col] = [lower_b, upper_b]

        mean_use = df[col][(df[col] > bounds[col][0]) & (df[col] < bounds[col][1])].mean()
        df[col].where((df[col] > bounds[col][0]) & (df[col] < bounds[col][1]), mean_use, inplace=True)

        return df

    # fill nan values with values
    def filling_nan(self, df: pd.DataFrame, cols, value) -> pd.DataFrame:
        for col in cols:
            df[col].fillna(value, inplace=True)

    # drop duplicate
    def drop_duplicate(self, df: pd.DataFrame, column) -> pd.DataFrame:
        df = df.drop_duplicates(subset=[column])

        return df

    def make_datefeatures(self, df: pd.DataFrame, column) -> pd.DataFrame:
        df['months'] = df[column].dt.month
        df['Years'] = df[column].dt.year
        df['DayOfYear'] = df[column].dt.dayofyear
        df['WeekOfYear'] = df[column].dt.weekofyear
        df['dayType'] = df['DayOfWeek'].apply(lambda x: 1 if x > 5 else 0)
        return df

    def label_encoder(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_features = df.select_dtypes(include='object').columns.tolist()
        for i in categorical_features:
            df[categorical_features] = df[categorical_features].apply(lambda x: pd.factorize(x)[0])

        return df

    def scale_data(self, df: pd.DataFrame) -> pd.DataFrame:

        scaler = MinMaxScaler()
        df[:] = scaler.fit_transform(df[:])
        return df, scaler
