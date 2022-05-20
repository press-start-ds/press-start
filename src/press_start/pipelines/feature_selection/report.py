import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Union, Optional
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import datapane as dp
import pandas as pd
import os
import itertools
import tempfile
from press_start.params import GeneralParams


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


def _get_confusion_matrix(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
    plot_title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    fig, ax = plt.subplots(figsize=(10, 9))
    y = df.iloc[:, 0]
    y_hat = df.iloc[:, 1:].idxmax(axis=1)
    labels = df.columns[1:]
    cf_matrix = confusion_matrix(y, y_hat)
    n_data = cf_matrix.sum()
    cm_values = np.vectorize(lambda v: "{:d}\n({:0.1f}%)".format(v, 100 * v / n_data))(
        cf_matrix
    )
    ax = sns.heatmap(cf_matrix, annot=cm_values, fmt="", cmap="Blues")

    ax.set_title(plot_title)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    fig.tight_layout()
    # general_params = GeneralParams(general_params_dict)
    return ax


def _get_classification_metrics(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, dict],
):
    y_hat = df.iloc[:, 1:].idxmax(axis=1)
    dict_report = classification_report(
        y_true=df.iloc[:, 0], y_pred=y_hat, output_dict=True
    )
    return pd.DataFrame.from_dict(dict_report)


def _get_classification_report(
    dict_df_predictions: Dict[str, pd.DataFrame],
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, dict],
) -> str:
    conf_matrices = []
    metrics = []
    labels = []
    for label, df_pred in dict_df_predictions.items():
        conf_matrices.append(
            _get_confusion_matrix(df_pred, params, general_params_dict)
        )
        metrics.append(
            _get_classification_metrics(df_pred, params, general_params_dict)
        )
        labels.append(label)

    # TODO: Use `to_string` method from datapane instead of files
    with tempfile.TemporaryDirectory() as folder:
        path_file = os.path.join(folder, "report.html")
        dp.Report(
            dp.Page(
                title="Confusion matrices",
                blocks=itertools.chain.from_iterable(
                    (f"### {label}", dp.Plot(cm))
                    for cm, label in zip(conf_matrices, labels)
                ),
            ),
            dp.Page(
                title="Classification metrics",
                blocks=itertools.chain.from_iterable(
                    (f"### {label}", dp.Plot(cm)) for cm, label in zip(metrics, labels)
                ),
            ),
        ).save(
            path=path_file, formatting=dp.ReportFormatting(width=dp.ReportWidth.NARROW)
        )
        with open(path_file, "r") as f:
            html_content = f.read()

        return html_content


def report_feat_selection_metrics(
    k_best: pd.DataFrame,
    rfe: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, dict],
) -> str:
    return _get_classification_report(
        {
            "sklearn SelectKBest": k_best,
            "sklearn Recursive Feature Elimination (RFE)": rfe,
        },
        params,
        general_params_dict,
    )
