import pandas as pd
from press_start.pipelines.feature_selection.nodes import get_metrics
from press_start.params import GeneralParams


def test_get_metrics(df_dataset, df_metrics):
    general_params = GeneralParams({"column_target": "target"})
    df_res = get_metrics(df_dataset, {}, general_params)
    pd.testing.assert_frame_equal(df_res, df_metrics)
