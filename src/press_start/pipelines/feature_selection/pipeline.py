from kedro.pipeline import Pipeline, node

from .nodes import feat_selection_k_best


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=feat_selection_k_best,
                inputs=[
                    "dev_dataset",
                    "params:feat_selection_k_best",
                    "params:general",
                ],
                outputs="feat_selection_k_best",
                name="feat_selection_k_best",
                tags="feature_selection",
            ),
        ]
    )
