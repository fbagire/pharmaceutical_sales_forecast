import pandas as pd
import numpy as np
from logger_comb import logger

def load_data(filepath):
    df = pd.read_csv(filepath)
    logger.info('Successfully loaded datasets')
    return df

class clean_data():
    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df

    #fixing outliers
    def fix_outliers(self, df: pd.DataFrame, column) -> pd.DataFrame:
        df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
        logger.info('successfull removing outliers')

        return df[column]

    # converting columns to string
    def convert_to_string(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        for col in columns:
            df[col] = df[col].astype("string")
            logger.info('successfully Converting to string')

    # converting column to int
    def convert_to_int(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        for col in columns:
            df[col] = df[col].astype("int64")
            logger.info(' successfully converting to int')

    # handling categorial and numeric columns by filling with mean and median and model
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

    # fill nan values with values
    def filling_nan(self, df: pd.DataFrame, cols, value) -> pd.DataFrame:
        for col in cols:
            df[col].fillna(value, inplace=True)

    # unique values in columns
    def unique_values(self, df: pd.DataFrame, column) -> pd.DataFrame:
        unique_values = df[column].unique()
        return unique_values

    # drop rows with nan values:
    def drop_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(inplace=True)
        logger.info('dropped column with nan values')

    def drop_cols(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        df.drop(cols, axis=1, inplace=True)
        logger.info("successful deleted a column")
        return df

    # drop duplicate
    def drop_duplicate(self, df: pd.DataFrame, column) -> pd.DataFrame:
        df = df.drop_duplicates(subset=[column])
        return df

    def transform_columns(self, df: pd.DataFrame, column) -> pd.DataFrame:
        df['Days'] = df[column].dt.day
        df['months'] = df[column].dt.month
        df['Years'] = df[column].dt.year
        df['DayOfYear'] = df[column].dt.dayofyear
        df['WeekOfYear'] = df[column].dt.weekofyear
