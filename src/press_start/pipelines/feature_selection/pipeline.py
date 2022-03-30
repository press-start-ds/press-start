from kedro.pipeline import Pipeline, node

from .nodes import (
    feature_selection_corr
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=feature_selection_corr,
            inputs=["input_dataset", "params:feature_selection"],
            outputs="feature_selection_corr",
            name="feature_selection_corr",
            tags="feature_selection"
        ),
    ])
