from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from press_start.nodes.sklearn.cluster import KMeans
from press_start.nodes.attributes import drop_columns
from .nodes import load_iris_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_iris_dataset,
                inputs=None,
                outputs="dataset_iris",
                name="load_iris",
                tags="data_loading",
            ),
            node(
                func=drop_columns,
                inputs=["dataset_iris", "params:drop_columns"],
                outputs="X_iris",
                name="select_attributes",
                tags="data_loading",
            ),
            node(
                func=KMeans,
                inputs=["X_iris", "params:kmeans"],
                outputs=["example_output_iris", "example_model_iris"],
                name="kmeans",
                tags="clustering",
            ),
        ],
        namespace="example_clustering",
    )
