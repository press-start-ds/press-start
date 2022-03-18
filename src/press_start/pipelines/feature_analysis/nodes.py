from pandas_profiling import ProfileReport
import pandas as pd


def pandas_profile(df: pd.DataFrame):
    profile = ProfileReport(df)
    return profile.to_html()
