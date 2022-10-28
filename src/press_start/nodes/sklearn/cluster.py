import sys
from press_start.nodes.utils import register_external_node
from typing import Callable
import pandas as pd


_SKLEARN_MODULE = "sklearn.cluster"
_SKLEARN_CLASSES = (
    "AffinityPropagation",
    "AgglomerativeClustering",
    "Birch",
    "DBSCAN",
    "FeatureAgglomeration",
    "KMeans",
    "BisectingKMeans",
    "MiniBatchKMeans",
    "MeanShift",
    "OPTICS",
    "SpectralClustering",
    "SpectralBiclustering",
    "SpectralCoclustering",
)

AffinityPropagation: Callable
AgglomerativeClustering: Callable
Birch: Callable
DBSCAN: Callable
FeatureAgglomeration: Callable
KMeans: Callable
BisectingKMeans: Callable
MiniBatchKMeans: Callable
MeanShift: Callable
OPTICS: Callable
SpectralClustering: Callable
SpectralBiclustering: Callable
SpectralCoclustering: Callable


def register_cluster_model(base_model):
    def model_fit(df, parameters):
        model = base_model(**parameters)
        model.fit(df)
        if hasattr(model, "labels_"):
            y_pred = model.labels_.astype(int)
        else:
            y_pred = model.predict(df)
        return pd.Series(y_pred, name="cluster"), base_model

    return model_fit


register_external_node(
    _SKLEARN_MODULE, _SKLEARN_CLASSES, sys.modules[__name__], register_cluster_model
)
