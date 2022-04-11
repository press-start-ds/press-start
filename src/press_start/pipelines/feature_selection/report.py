import scikitplot as skplt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Union
from sklearn.ensemble import RandomForestClassifier
from press_start.utils import GeneralParams
import pandas as pd


def get_metrics(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> pd.DataFrame:
    general_params = GeneralParams(general_params_dict)
    df_tr = df[lambda df: df._is_training].drop("_is_training", axis=1)
    df_ts = df[lambda df: ~df._is_training].drop("_is_training", axis=1)

    X_tr, y_tr = (
        df_tr.drop(general_params.column_target),
        df_tr[general_params.column_target],
    )
    X_ts, y_ts = (
        df_ts.drop(general_params.column_target),
        df_ts[general_params.column_target],
    )

    clf = RandomForestClassifier()
    clf.fit(X_tr, y_tr)
    pd.DataFrame((y_ts, clf.predict_proba(X_ts)), index=["y_true", "y_hat_prob"])


def generate_report(*args):
    fig_conf_matrix = plt.figure(figsize=(15, 6))
    skplt.metrics.plot_confusion_matrix(y_ts, y_hat, title="Confusion Matrix")

    return fig_conf_matrix
