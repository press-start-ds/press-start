from kedro.pipeline import Pipeline, node

from .nodes import data_split, category_encoder


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=category_encoder,
                inputs=[
                    "input_dataset",
                    "params:category_encoder",
                    "params:general",
                ],
                outputs=["category_encoder", "numerical_dataset"],
                name="category_encoder",
                tags="data_split",
            ),
            node(
                func=data_split,
                inputs=["input_dataset", "params:data_split", "params:general"],
                outputs=["dev_dataset", "holdout_dataset"],
                name="data_split",
                tags="data_split",
            ),
        ]
    )
