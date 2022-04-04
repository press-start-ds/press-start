from kedro.pipeline import Pipeline, node

from .nodes import (
    data_split
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=data_split,
            inputs=["input_dataset", "params:data_split", "params:general"],
            outputs=["dev_dataset", "holdout_dataset"],
            name="data_split",
            tags="data_split"
        ),
    ])
