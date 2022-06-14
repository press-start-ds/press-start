from kedro.pipeline import Pipeline, node

from .nodes import load_example_dataset


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_example_dataset,
                inputs=None,
                outputs="nlp_example_dataset",
                name="load_nlp_example_dataset",
                tags="nlp_visualization",
            ),
        ]
    )
