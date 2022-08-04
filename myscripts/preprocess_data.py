import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from myscripts.logger_comb import logger


class clean_data():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # fixing outliers
    def fix_outliers(self, df: pd.DataFrame, column) -> pd.DataFrame:
        for col in column:
            df[col] = np.where(df[col] > df[col].quantile(0.95), df[col].mean(), df[col])
        logger.info('successfull removing outliers')

        return df

    # handling categorial and numeric columns by filling with mean and median and mode
    def handling_missing(self, df: pd.DataFrame) -> pd.DataFrame:

        # numeric column
        df_num = df.select_dtypes(include=["float", "int"])
        normal_dist = []
        skewed = []
        for i in df_num.columns:
            # checking for skewness
            if 0.5 > df_num[i].skew() > -0.5:
                normal_dist.append(i)
            else:
                skewed.append(i)
        # for normal distribution values fill with median
        for t in normal_dist:
            df[t].fillna(df[t].median(), inplace=True)
        # for skewed fill with mean
        for j in skewed:
            df[j].fillna(df[j].mean(), inplace=True)

        # for categorical fill with model
        df_cat = df.select_dtypes(include=["object"])
        for n in df_cat.columns:
            df[n].fillna(df[n].mode()[0], inplace=True)
        logger.info('successful fixed missing values')

        return df

    # fill nan values with values
    def filling_nan(self, df: pd.DataFrame, cols, value) -> pd.DataFrame:
        for col in cols:
            df[col].fillna(value, inplace=True)
        logger.info('successful filled nan with input value')

    # drop duplicate
    def drop_duplicate(self, df: pd.DataFrame, column) -> pd.DataFrame:
        df = df.drop_duplicates(subset=[column])
        logger.info('successful dropped duplicates')

        return df

    def transform_columns(self, df: pd.DataFrame, column) -> pd.DataFrame:
        df['Days'] = df[column].dt.day
        df['months'] = df[column].dt.month
        df['Years'] = df[column].dt.year
        df['DayOfYear'] = df[column].dt.dayofyear
        df['WeekOfYear'] = df[column].dt.weekofyear

    def label_encoder(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_features = df.select_dtypes(include='object').columns.tolist()
        for i in categorical_features:
            df[categorical_features] = df[categorical_features].apply(lambda x: pd.factorize(x)[0])

        logger.info('successful convert to numeric')

        return df

    def scalling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        df[:] = scaler.fit_transform(df[:])
        logger.info("successful scaling data")
        return df
