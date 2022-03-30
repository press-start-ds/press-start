from kedro.pipeline import Pipeline, node

from .nodes import pandas_profile


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=pandas_profile,
            inputs=["input_dataset", "params:feature_analysis"],
            outputs="pandas_profile",
            name="pandas_profile",
        ),
    ])
