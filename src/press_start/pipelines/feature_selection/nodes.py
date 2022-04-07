import pandas as pd
from typing import Dict, Union
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from press_start.utils import GeneralParams


def feat_selection_k_best(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
):
    general_params = GeneralParams(general_params_dict)
    df_target = df[general_params.column_target]
    df_feat = df.drop(general_params.column_target, axis=1)
    model = SelectKBest(chi2, k=params.get("k", 1)).fit(df_feat, df_target)
    selected_cols = model.get_feature_names_out()
    return pd.concat((df_feat[selected_cols], df_target), axis=1)
