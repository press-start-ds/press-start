from kedro.pipeline import Pipeline, node

from .nodes import data_split, category_encoder


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=data_split,
                inputs=dict(
                    df="input_dataset",
                    params="params:data_split",
                    general_params_dict="params:general",
                ),
                outputs=["dev_dataset", "holdout_dataset"],
                name="data_split",
                tags="data_split",
            ),
            node(
                func=category_encoder,
                inputs=dict(
                    df="dev_dataset",
                    params="params:category_encoder",
                    general_params_dict="params:general",
                ),
                outputs=["category_encoder", "dev_dataset_numerical"],
                name="category_encoder",
                tags="data_split",
            ),
        ]
    )
