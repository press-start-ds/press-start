import pandas as pd


def select_columns(df: pd.DataFrame, columns):
    return df.loc[columns]


def drop_columns(df: pd.DataFrame, columns):
    if isinstance(columns, str):
        columns = (columns,)
    return df.drop(columns, axis=1)
