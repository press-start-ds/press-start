from kedro.pipeline import Pipeline, node

from .nodes import (
    pandas_profile, missing_no
)


def create_pipeline(params, **kwargs):
    nodes = dict(
        pandas_profile=node(
            func=pandas_profile,
            inputs=["input_dataset", "params:feature_analysis"],
            outputs="pandas_profile",
            name="pandas_profile",
        ),
        # missing_no=node(
        #     func=missing_no,
        #     inputs=["input_dataset", "params:feature_analysis"],
        #     outputs="missing_no",
        #     name="missing_no",
        # ),
    )
    return Pipeline([
        v for k, v in nodes.items()
        if params.get(f'feature_analysis.{k}._run', False)
    ])
