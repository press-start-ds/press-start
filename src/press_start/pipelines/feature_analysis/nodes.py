import pandas as pd
from typing import Dict, Union


def pandas_profiling(df: pd.DataFrame, params: Dict[str, Union[float, Dict]]) -> str:
    """Runs pandas profilling

    Args:
        df: Input dataframe
        params: Paramters defined in conf/*/parameters.yml

    Returns:
        str: Pandas profilling report as html
    """
    if params.get("_run", False):
        from pandas_profiling import ProfileReport

        sample_size = params.get("sample_fraction", 1.0)
        profile = ProfileReport(df.sample(frac=sample_size), **params.get("params", {}))
        return profile.to_html()

    return "Activate pandas profiling to get the repport."
