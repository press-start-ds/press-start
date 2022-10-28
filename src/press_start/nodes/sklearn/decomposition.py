import sys
from press_start.nodes.utils import register_external_node
from typing import Callable
import pandas as pd


_SKLEARN_MODULE = "sklearn.decomposition"
_SKLEARN_CLASSES = (
    "DictionaryLearning",
    "FactorAnalysis",
    "FastICA",
    "IncrementalPCA",
    "KernelPCA",
    "LatentDirichletAllocation",
    "MiniBatchDictionaryLearning",
    "MiniBatchSparsePCA",
    "NMF",
    "MiniBatchNMF",
    "PCA",
    "SparsePCA",
    "SparseCoder",
    "TruncatedSVD",
)

DictionaryLearning: Callable
FactorAnalysis: Callable
FastICA: Callable
IncrementalPCA: Callable
KernelPCA: Callable
LatentDirichletAllocation: Callable
MiniBatchDictionaryLearning: Callable
MiniBatchSparsePCA: Callable
NMF: Callable
MiniBatchNMF: Callable
PCA: Callable
SparsePCA: Callable
SparseCoder: Callable
TruncatedSVD: Callable


def register_decomposition_model(base_model):
    def model_fit(df, parameters):
        model = base_model(**parameters)
        model.fit(df)
        return (pd.DataFrame(model.transform(df), columns=["x1", "x2"]), base_model)

    return model_fit


register_external_node(
    _SKLEARN_MODULE,
    _SKLEARN_CLASSES,
    sys.modules[__name__],
    register_decomposition_model,
)
