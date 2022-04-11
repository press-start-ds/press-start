import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Union

from press_start.utils import GeneralParams


def data_split(
    df: pd.DataFrame,
    params: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
):
    def _stratify(df):
        if params.get("stratify", False):
            return df[general_params.column_target]

    general_params = GeneralParams(general_params_dict)
    val_size = params.get("val_size", 0)
    test_size = params.get("test_size", 0)
    if isinstance(val_size, float):
        val_size = round(df.shape[0] * val_size)
    if isinstance(test_size, float):
        test_size = round(df.shape[0] * test_size)
    shuffle = params.get("shuffle", True)

    df_dev, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=general_params.prng_seed,
        shuffle=shuffle,
        stratify=_stratify(df),
    )
    df_train, df_val = train_test_split(
        df_dev,
        test_size=val_size,
        random_state=general_params.prng_seed,
        shuffle=shuffle,
        stratify=_stratify(df_dev),
    )
    df_dev = pd.concat(
        (
            df_train.assign(_is_training=True),
            df_val.assign(_is_training=False),
        )
    )
    return df_dev, df_test
