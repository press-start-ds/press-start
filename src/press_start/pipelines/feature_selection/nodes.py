import pandas as pd
import numpy as np
from typing import Dict, Union
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from press_start.utils import GeneralParams


def feat_selection_corr(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
):
    general_params = GeneralParams(general_params_dict)
    max_corr = params.get('max_corr', 1)
    df_target = df[general_params.column_target]
    df_feat = df.drop(general_params.column_target, axis=1)
    corr = df_feat.corr()
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    df_feat = df_feat.loc[:, (upper_tri > max_corr).any()]
    return pd.concat((df_feat, df_target), axis=1)

def feat_selection_k_best(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
):
    SelectKBest(chi2, k=2).fit_transform(X, y)

