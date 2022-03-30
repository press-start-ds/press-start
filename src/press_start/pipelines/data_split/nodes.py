import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict


def data_split(df: pd.DataFrame, params: Dict[str, Dict]):
    params = params.get('data_split', {})
    if 'dev_size' in params:
        dev_size = params['dev_size']
        assert (dev_size >= 0) and (dev_size < df.shape[0] - 1), (
            "data_split.dev_size must be in [0, n_rows - 1]"
        )
    else:
        dev_frac = params.get('dev_fraction', 0)
        assert (dev_frac >= 0) and (dev_frac < 1), (
            "data_split.dev_fraction must be in [0.0, 1.0)."
        )

    return pd.concat((df_feat, df_target), axis=1)

