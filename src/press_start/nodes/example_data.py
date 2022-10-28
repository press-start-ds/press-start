from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


def load_iris_dataset():
    dataset = load_iris()
    data = np.hstack((dataset.data, dataset.target[:, None]))
    df = pd.DataFrame(data, columns=dataset.feature_names + ["target"])
    df = df.assign(
        target=lambda df: df.target.replace(
            {i: name for i, name in enumerate(dataset.target_names)}
        )
    )
    return df
