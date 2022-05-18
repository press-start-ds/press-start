from kedro.pipeline import Pipeline, node

from .report import get_metrics, report_confusion_matrix_feat_selection
from .nodes import feat_selection_k_best


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=feat_selection_k_best,
                inputs=dict(
                    df="dev_dataset",
                    params="params:feat_selection_k_best",
                    general_params_dict="params:general",
                ),
                outputs="feat_selection_k_best",
                name="feat_selection_k_best",
                tags="feature_selection",
            ),
            node(
                func=get_metrics,
                inputs=dict(
                    df="feat_selection_k_best",
                    params="params:feat_selection_metrics",
                    general_params_dict="params:general",
                ),
                outputs="feat_selection_k_best_metrics",
                name="generate_k_best_metrics",
                tags="feature_selection",
            ),
            node(
                func=report_confusion_matrix_feat_selection,
                inputs=dict(
                    k_best="feat_selection_k_best_metrics",
                    params="params:feat_selection_metrics",
                    general_params_dict="params:general",
                ),
                outputs="feat_selection_report",
                name="generate_feat_selection_report",
                tags="feature_selection",
            ),
        ]
    )
