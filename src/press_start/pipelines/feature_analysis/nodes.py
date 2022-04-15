import pandas as pd
from typing import Dict


def pandas_profile(df: pd.DataFrame, params: Dict[str, Dict]) -> str:
    """Runs pandas profilling

    Args:
        df: Input dataframe
        params: Paramters defined in conf/*/parameters.yml

    Returns:
        str: Pandas profilling report as html
    """
    from pandas_profiling import ProfileReport

    params = params.get("pandas_profile", {})

    sample_size = params.get("sample_fraction", 1)
    profile = ProfileReport(df.sample(frac=sample_size), **params.get("params", {}))
    return profile.to_html()
