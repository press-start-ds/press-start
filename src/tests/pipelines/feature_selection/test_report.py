import pandas as pd
import numpy as np
import matplotlib
from press_start.pipelines.feature_selection.report import (
    _get_classification_metrics,
    _get_confusion_matrix,
    _get_classification_report,
)

np.random.seed(123)


def test_get_confusion_matrix(df_metrics):
    ax = _get_confusion_matrix(df_metrics, {}, {"column_target": "target"})
    assert isinstance(ax, matplotlib.axes.Axes)


def test_get_classification_metrics(df_metrics):
    df_exp_report = pd.DataFrame.from_dict(
        {
            "0.0": {
                "precision": 0.5,
                "recall": 1.0,
                "f1-score": 0.6666666666666666,
                "support": 1,
            },
            "1.0": {
                "precision": 0.3333333333333333,
                "recall": 1.0,
                "f1-score": 0.5,
                "support": 2,
            },
            "2.0": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 5},
            "accuracy": 0.375,
            "macro avg": {
                "precision": 0.27777777777777773,
                "recall": 0.6666666666666666,
                "f1-score": 0.38888888888888884,
                "support": 8,
            },
            "weighted avg": {
                "precision": 0.14583333333333331,
                "recall": 0.375,
                "f1-score": 0.20833333333333331,
                "support": 8,
            },
        }
    )
    df_report = _get_classification_metrics(df_metrics, {}, {})
    pd.testing.assert_frame_equal(df_exp_report, df_report)


def test_get_classifiction_report(df_metrics):
    str_report = _get_classification_report({"method_name": df_metrics}, {}, {})
    assert "method_name" in str_report
