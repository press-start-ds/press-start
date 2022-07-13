from kedro.pipeline import Pipeline, node

from .nodes import pandas_profiling


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=pandas_profiling,
                inputs=dict(df="dev_dataset", params="params:pandas_profiling"),
                outputs="pandas_profile",
                name="pandas_profiling",
            ),
        ]
    )
