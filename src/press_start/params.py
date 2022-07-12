from typing import Dict, Any


class GeneralParams:
    def __init__(self, params: Dict[str, Any]):
        self.columns_to_keep = params.get("columns_to_keep")
        self.columns_to_drop = params.get("columns_to_drop")
        self.column_target = params.get("column_target")
        self.columns_categorical = params.get("columns_categorical", [])
        self.prng_seed = params.get("prng_seed", 12345)


class Params:
    def __init__(self, params: Dict[str, Any]):
        self._run = params.get("_run", False)


class ParamsPandasProfiling(Params):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.sample_fraction = params.get("sample_fraction", 1.0)
        self.params = params.get(
            "params", dict(samples=dict(head=5, tail=5), interactions=None)
        )


class ParamsFeatSelectionKBest(Params):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.k = params.get("k", 3)


class ParamsFeatSelectionRFE(Params):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.n_features_to_select = params.get("n_features_to_select", None)


class ParamsFeatSelectionMetrics(Params):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.k = params.get("k", 3)


class ParamsDataSplit(Params):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.val_size = params.get("val_size", 0.15)
        self.test_size = params.get("test_size", 0.15)
        self.stratify = params.get("stratify", True)


class ParamsCategoryEncoder(Params):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)


class ParamsNLPViz(Params):
    DIMENSIONALITY_REDUCTION_METHODS = {"umap"}

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.column_corpus = params.get("column_corpus", "doc")
        self.column_id = params.get("column_id", "id_doc")
        self.column_category = params.get("column_category", None)
        self.dim_reduction_method = params.get("dim_reduction_method", "umap")
        if (
            self.dim_reduction_method
            not in ParamsNLPViz.DIMENSIONALITY_REDUCTION_METHODS
        ):
            raise Exception(
                "The dimension reduction method was not found. It must be one of:"
                f"{'.'.join(ParamsNLPViz.DIMENSIONALITY_REDUCTION_METHODS)}"
            )
        self.dim_reduction_args = params.get("dim_reduction_args", {})
        self.data_clustering_method = params.get("data_clustering_method", None)
        self.data_clustering_args = params.get("data_clustering_args", {})
