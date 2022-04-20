from press_start.pipelines.data_split.nodes import category_encoder
import pandas as pd
import numpy as np


def test_category_encoder(df_categorical):
    enc, df_numeric = category_encoder(
        df_categorical,
        {},
        {"columns_categorical": ["buying", "maint"], "column_target": "class"},
    )
    df_exp = pd.DataFrame.from_dict(
        {
            "buying_high": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0},
            "buying_low": {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0},
            "buying_med": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0},
            "maint_low": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0},
            "maint_med": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0},
            "year": {0: 2007, 1: 2010, 2: 2015, 3: 1989, 4: 2008},
            "class": {0: "unacc", 1: "good", 2: "acc", 3: "unacc", 4: "unacc"},
        }
    )
    df_numeric.to_csv("/tmp/df_numeric.csv")
    df_exp.to_csv("/tmp/df_exp.csv")

    enc_categories_exp = np.array(["high", "low", "med", "low", "med"])
    pd.testing.assert_frame_equal(df_exp, df_numeric, check_like=True)
    np.testing.assert_array_equal(np.concatenate(enc.categories_), enc_categories_exp)
