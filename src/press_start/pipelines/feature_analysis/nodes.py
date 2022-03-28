import pandas as pd


def pandas_profile(df: pd.DataFrame, params):
    from pandas_profiling import ProfileReport

    params = params.get('pandas_profile', {})

    sample_size = params.get('sample_fraction', 1)
    profile = ProfileReport(
        df.sample(frac=sample_size), **params.get('params', {})
    )
    return profile.to_html()


def missing_no(df: pd.DataFrame, params):
    import missingno as msno

    return msno.matrix(df).figure
