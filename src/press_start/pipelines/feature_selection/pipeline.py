from kedro.pipeline import Pipeline, node

from .nodes import (
    feature_selection_corr
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=feature_selection_corr,
            inputs=["dev_dataset", "params:feature_selection"],
            outputs=["feat_selection_correlation", "feat_selection_corr_report"],
            name="feat_selection_correlation",
            tags="feature_selection"
        ),
        node(
            func=feature_selection_corr,
            inputs=["dev_dataset", "params:feature_selection"],
            outputs=["feat_selection_rec_elim", "feat_selection_rec_elim_report"],
            name="feat_selection_recursive_elimination",
            tags="feature_selection"
        ),
    ])
