import scikitplot as skplt
import matplotlib.pyplot as plt
from typing import Dict, Union
from sklearn.ensemble import RandomForestClassifier
from press_start.utils import GeneralParams
import pandas as pd
import numpy as np


def get_metrics(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> pd.DataFrame:
    general_params = GeneralParams(general_params_dict)
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
    return df_ts.T.sample(0).to_frame()


def get_confusion_matrix(
    df_dict: Dict[str, pd.DataFrame],
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> plt.figure:
    general_params = GeneralParams(general_params_dict)
    fig_conf_matrix, axs = plt.subplots(len(df_dict), 1)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]
    for (method_name, df), ax in zip(df_dict.items(), axs):
        y_hat = df.iloc[:, 1:].idxmax(axis=1)
        skplt.metrics.plot_confusion_matrix(
            df[general_params.column_target], y_hat, title="Confusion Matrix", ax=ax
        )
        return fig_conf_matrix


def report_confusion_matrix_feat_selection(
    k_best: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> plt.figure:
    return get_confusion_matrix(
        {"sklearn SelectKBest": k_best}, params, general_params_dict
    )
