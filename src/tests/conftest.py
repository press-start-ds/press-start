import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def df_categorical():
    return pd.DataFrame.from_dict(
        {
            "buying": {0: "med", 1: "low", 2: "low", 3: "high", 4: "med"},
            "maint": {0: "med", 1: "med", 2: "low", 3: "med", 4: "low"},
            "year": {0: 2007, 1: 2010, 2: 2015, 3: 1989, 4: 2008},
            "class": {0: "unacc", 1: "good", 2: "acc", 3: "unacc", 4: "unacc"},
        }
    )


@pytest.fixture
def df_metrics(scope="session"):
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
