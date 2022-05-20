import pandas as pd
from typing import Dict, Union, Tuple
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from press_start.params import (
    GeneralParams,
    ParamsFeatSelectionKBest,
    ParamsFeatSelectionRFE,
)


def get_metrics(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params: GeneralParams,
) -> pd.DataFrame:
    df_tr = df[lambda df: df["_is_training"]].drop("_is_training", axis=1)
    df_ts = df[lambda df: ~df["_is_training"]].drop("_is_training", axis=1)

    if not df.empty:
        X_tr, y_tr = (
            df_tr.drop(general_params.column_target, axis=1),
            df_tr[general_params.column_target],
        )
        X_ts, y_ts = (
            df_ts.drop(general_params.column_target, axis=1),
            df_ts[general_params.column_target],
        )

        clf = RandomForestClassifier(random_state=general_params.prng_seed)
        clf.fit(X_tr, y_tr)
        y_hat_prob = clf.predict_proba(X_ts)
        y_hat_prob_names = clf.classes_
        df_prediction = pd.concat(
            (
                y_ts,
                pd.DataFrame(y_hat_prob, columns=y_hat_prob_names, index=y_ts.index),
            ),
            axis=1,
        )

        return df_prediction
    return pd.DataFrame()


def feat_selection_k_best(
    df: pd.DataFrame,
    params_dict: Dict[str, Union[int, float]],
    metric_params_dict: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = ParamsFeatSelectionKBest(params_dict)
    df_result: pd.DataFrame
    if params._run:
        general_params = GeneralParams(general_params_dict)
        df_feat = df.drop(general_params.column_target, axis=1)
        df_target = df[general_params.column_target]
        df_is_training = df["_is_training"]

        selector = SelectKBest(chi2, k=params.k).fit(
            df_feat[df_is_training], df_target[df_is_training]
        )
        selected_cols = selector.get_feature_names_out()
        df_result = pd.concat(
            (df_feat[selected_cols], df_is_training, df_target), axis=1
        )
        df_metrics = get_metrics(df_result, metric_params_dict, general_params)
    else:
        df_result = pd.DataFrame()
        df_metrics = pd.DataFrame()
    return df_result, df_metrics


def feat_selection_rfe(
    df: pd.DataFrame,
    params_dict: Dict[str, Union[int, float]],
    metric_params_dict: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = ParamsFeatSelectionRFE(params_dict)
    df_result: pd.DataFrame
    if params._run:
        general_params = GeneralParams(general_params_dict)
        df_feat = df.drop(general_params.column_target, axis=1)
        df_target = df[general_params.column_target]
        df_is_training = df["_is_training"]
        selector = RFE(
            RandomForestClassifier(), n_features_to_select=params.n_features_to_select
        )
        selector.fit(df_feat[df_is_training], df_target[df_is_training])
        selected_cols = selector.get_feature_names_out()
        df_result = pd.concat(
            (df_feat[selected_cols], df_is_training, df_target), axis=1
        )
        df_metrics = get_metrics(df_result, metric_params_dict, general_params)
    else:
        df_result = pd.DataFrame()
        df_metrics = pd.DataFrame()
    return df_result, df_metrics
