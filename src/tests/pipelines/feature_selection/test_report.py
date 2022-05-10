import pytest
import pandas as pd
import numpy as np
import matplotlib
from sklearn.datasets import make_classification
from press_start.pipelines.feature_selection.report import (
    get_classification_metrics,
    get_metrics,
    get_confusion_matrix,
    get_classification_report,
)

np.random.seed(123)


@pytest.fixture
def df_metrics():
    return pd.DataFrame(
        [
            [2.0, 0.45, 0.47, 0.08],
            [0.0, 0.43, 0.39, 0.18],
            [1.0, 0.13, 0.81, 0.06],
            [2.0, 0.34, 0.47, 0.19],
            [1.0, 0.23, 0.65, 0.12],
            [2.0, 0.2, 0.69, 0.11],
            [2.0, 0.18, 0.43, 0.39],
            [2.0, 0.52, 0.41, 0.07],
        ],
        columns=["target", 0.0, 1.0, 2.0],
        index=range(12, 20),
    )


@pytest.fixture
def df_dataset():
    n_samples = 20
    n_features = 3
    n_classes = 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=0,
        random_state=123,
    )
    return (
        pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1))
        .set_axis([f"f{i}" for i in range(1, 4)] + ["target"], axis=1)
        .assign(_is_training=np.random.random(n_samples) > 0.4)
        .sort_values("_is_training", ascending=False)
        .reset_index(drop=True)
    )


def test_get_metrics(df_dataset, df_metrics):
    df_res = get_metrics(df_dataset, {}, {"column_target": "target"})
    pd.testing.assert_frame_equal(df_res, df_metrics)


def test_get_confusion_matrix(df_metrics):
    ax = get_confusion_matrix(df_metrics, {}, {"column_target": "target"})
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
    df_report = get_classification_metrics(df_metrics, {}, {})
    pd.testing.assert_frame_equal(df_exp_report, df_report)


def test_get_classifiction_report(df_metrics):
    str_report = get_classification_report({"method_name": df_metrics}, {}, {})
    assert "method_name" in str_report