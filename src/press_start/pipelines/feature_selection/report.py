import scikitplot as skplt
import matplotlib
from typing import Dict, Union, Optional
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from press_start.utils import GeneralParams
import datapane as dp
import pandas as pd
import os
import tempfile


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
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
    plot_title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    # general_params = GeneralParams(general_params_dict)
    y_hat = df.iloc[:, 1:].idxmax(axis=1)
    return skplt.metrics.plot_confusion_matrix(df.iloc[:, 0], y_hat, title=plot_title)


def report_confusion_matrix_feat_selection(
    k_best: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, dict],
) -> str:
    return get_classification_report(
        {"sklearn SelectKBest": k_best}, params, general_params_dict
    )


def get_classification_metrics(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, dict],
):
    y_hat = df.iloc[:, 1:].idxmax(axis=1)
    dict_report = classification_report(
        y_true=df.iloc[:, 0], y_pred=y_hat, output_dict=True
    )
    return pd.DataFrame.from_dict(dict_report)


def get_classification_report(
    dict_df_predictions: Dict[str, pd.DataFrame],
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, dict],
) -> str:
    conf_matrices = []
    metrics = []
    labels = []
    for label, df_pred in dict_df_predictions.items():
        conf_matrices.append(get_confusion_matrix(df_pred, params, general_params_dict))
        metrics.append(get_classification_metrics(df_pred, params, general_params_dict))
        labels.append(label)

    # TODO: Use `to_string` method from datapane instead of files
    with tempfile.TemporaryDirectory() as folder:
        path_file = os.path.join(folder, "report.html")
        dp.Report(
            dp.Page(
                title="Confusion matrices",
                blocks=[
                    dp.Plot(cm, label=label) for cm, label in zip(conf_matrices, labels)
                ],
            ),
            dp.Page(
                title="Classification metrics",
                blocks=[dp.Plot(cm, label=label) for cm, label in zip(metrics, labels)],
            ),
        ).save(path=path_file)
        with open(path_file, "r") as f:
            html_content = f.read()

        return html_content