import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Union

from press_start.utils import GeneralParams


def data_split(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
):
    general_params = GeneralParams(general_params_dict)
    test_size: Union[int, float]
    if "dev_size" in params:
        test_size = int(params["dev_size"])
        assert (test_size >= 0) and (
            test_size < df.shape[0] - 1
        ), "data_split.dev_size must be in [0, n_rows - 1]"
    else:
        test_size = params.get("dev_fraction", 0.)
        assert (test_size >= 0) and (
            test_size < 1
        ), "data_split.dev_fraction must be in [0.0, 1.0)."
    if params.get("stratify", False):
        stratify_data = df[general_params.column_target]
    else:
        stratify_data = None
    return train_test_split(
        df,
        test_size=test_size,
        random_state=general_params.prng_seed,
        shuffle=params.get("shuffle", True),
        stratify=stratify_data,
    )
