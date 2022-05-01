import scikitplot as skplt
import matplotlib.pyplot as plt
from typing import Dict, Union
from sklearn.ensemble import RandomForestClassifier
from press_start.utils import GeneralParams
import pandas as pd


def get_metrics(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
    data_name: str,
) -> pd.DataFrame:
    general_params = GeneralParams(general_params_dict)
    df_tr = df[lambda df: df._is_training].drop("_is_training", axis=1)
    df_ts = df[lambda df: ~df._is_training].drop("_is_training", axis=1)

    if not df.empty:
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
        df_pred = pd.DataFrame(
            (y_ts, clf.predict_proba(X_ts)), index=["y_true", "y_hat_prob"]
        )
        return df_pred
    return df_ts.T.sample(0)


def report_confusion_matrix(*args):
    # args: List of dataframes containing the y_true and y_hat_prob
    fig_conf_matrix, axs = plt.subplot(len(args), 1)
    for df, ax in zip(args, axs):
        df = args[0]
        # TODO: Use the ax to plot the conf. matrix
        skplt.metrics.plot_confusion_matrix(
            df.loc["y_true"], df.loc["y_hat_prob"], title="Confusion Matrix"
        )
        return fig_conf_matrix
