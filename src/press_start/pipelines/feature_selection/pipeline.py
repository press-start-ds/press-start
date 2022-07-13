from kedro.pipeline import Pipeline, node

from .report import report_feat_selection_metrics
from .nodes import feat_selection_k_best, feat_selection_rfe


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=feat_selection_k_best,
                inputs=dict(
                    df="dev_dataset_numerical",
                    params_dict="params:feat_selection_k_best",
                    metric_params_dict="params:feat_selection_metrics",
                    general_params_dict="params:general",
                ),
                outputs=["feat_selection_k_best", "feat_selection_k_best_metrics"],
                name="feat_selection_k_best",
                tags="feature_selection",
            ),
            node(
                func=feat_selection_rfe,
                inputs=dict(
                    df="dev_dataset_numerical",
                    params_dict="params:feat_selection_rfe",
                    metric_params_dict="params:feat_selection_metrics",
                    general_params_dict="params:general",
                ),
                outputs=["feat_selection_rfe", "feat_selection_rfe_metrics"],
                name="feat_selection_rfe",
                tags="feature_selection",
            ),
            node(
                func=report_feat_selection_metrics,
                inputs=dict(
                    k_best="feat_selection_k_best_metrics",
                    rfe="feat_selection_rfe_metrics",
                    params="params:feat_selection_metrics",
                    general_params_dict="params:general",
                ),
                outputs="feat_selection_report",
                name="generate_feat_selection_report",
                tags="feature_selection",
            ),
        ]
    )
