import pandas as pd


def pandas_profile(df: pd.DataFrame, params):
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df)
    return profile.to_html()


def missing_no(df: pd.DataFrame, params):
    import missingno as msno

    return msno.matrix(df).figure
