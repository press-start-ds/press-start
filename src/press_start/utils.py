from typing import Dict


class GeneralParams:
    def __init__(self, params: Dict[str, Dict]):
        self.columns_to_keep = params.get("columns_to_keep")
        self.columns_to_drop = params.get("columns_to_drop")
        self.column_target = params.get("column_target")
        self.prng_seed = params.get("prng_seed", 12345)
