from kedro.pipeline import Pipeline, node

from .nodes import (
    pandas_profile, missing_no
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=pandas_profile,
                inputs=["input_dataset", "params:feature_analysis"],
                outputs="pandas_profile",
                name="pandas_profile",
            ),
            node(
                func=missing_no,
                inputs=["input_dataset", "params:feature_analysis"],
                outputs="missing_no",
                name="missing_no",
            ),
        ]
    )
