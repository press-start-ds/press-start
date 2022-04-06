from kedro.pipeline import Pipeline, node

from .nodes import feat_selection_corr


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=feat_selection_corr,
            inputs=["dev_dataset", "params:feat_selection_corr", "params:general"],
            outputs=["feat_selection_correlation", "feat_selection_corr_report"],
            name="feat_selection_correlation",
            tags="feature_selection"
        ),
    ])
