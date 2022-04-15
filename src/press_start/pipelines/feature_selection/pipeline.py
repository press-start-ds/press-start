from kedro.pipeline import Pipeline, node

from .report import get_metrics
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
            # node(
            #     func=get_metrics,
            #     inputs=[
            #         "feat_selection_k_best",
            #         "params:feat_selection_metrics",
            #         "params:general",
            #     ],
            #     outputs="feat_selection_k_best_metrics",
            #     name="feat_selection_k_best_metrics",
            #     tags="feature_selection",
            # ),
        ]
    )
