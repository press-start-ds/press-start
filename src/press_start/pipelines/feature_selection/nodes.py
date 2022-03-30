import pandas as pd
import numpy as np
from typing import Dict


def feature_selection_corr(df: pd.DataFrame, params: Dict[str, Dict]):
    params = params.get('feature_selection_corr', {})
    columns_target = params.get('columns_target', [])
    max_corr = params.get('max_corr', 1)
    df_target = df[columns_target]
    df_feat = df.drop(columns_target, axis=1)
    corr = df_feat.corr()
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    df_feat = df_feat.loc[:, (upper_tri > max_corr).any()]
    return pd.concat((df_feat, df_target), axis=1)

