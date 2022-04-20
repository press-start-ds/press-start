from typing import Dict, Any


class GeneralParams:
    def __init__(self, params: Dict[str, Any]):
        self.columns_to_keep = params.get("columns_to_keep")
        self.columns_to_drop = params.get("columns_to_drop")
        self.column_target = params.get("column_target")
        self.columns_categorical = params.get("columns_categorical", [])
        self.prng_seed = params.get("prng_seed", 12345)
