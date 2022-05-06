import pandas as pd
from matplotlib.figure import Figure
from press_start.pipelines.feature_selection.report import (
    get_metrics,
    get_confusion_matrix,
)


df_metrics = pd.DataFrame.from_dict(
    {
        "target": {3: 1, 4: 0},
        0: {3: 0.79, 4: 0.33},
        1: {3: 0.21, 4: 0.67},
    }
)


def test_get_metrics():
    df_input = pd.DataFrame(
        [
            [9.8, 4.4, 1, True],
            [4.4, 9.5, 1, True],
            [4.0, 5.8, 0, True],
            [3.2, 5.2, 1, False],
            [4.7, 8.9, 0, False],
        ],
        columns=["f1", "f2", "target", "_is_training"],
    )
    df_res = get_metrics(df_input, {}, {"column_target": "target"})
    pd.testing.assert_frame_equal(df_res, df_metrics)


def test_get_confusion_matrix():
    fig = get_confusion_matrix({"Test": df_metrics}, {}, {"column_target": "target"})
    assert isinstance(fig, Figure)
