from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from press_start.nodes.reports import report_clustering_results

from press_start.nodes.sklearn.cluster import KMeans
from press_start.nodes.sklearn.decomposition import PCA
from press_start.nodes.attributes import drop_columns, select_columns
from press_start.nodes.example_data import load_iris_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_iris_dataset,
                inputs=None,
                outputs="example_dataset_iris",
                name="load_iris",
                tags="data_loading",
            ),
            node(
                func=drop_columns,
                inputs=["example_dataset_iris", "params:drop_columns"],
                outputs="X_iris",
                name="select_attributes",
                tags="data_loading",
            ),
            node(
                func=select_columns,
                inputs=["example_dataset_iris", "params:select_columns"],
                outputs="target",
                name="select_class",
                tags="data_loading",
            ),
            node(
                func=KMeans,
                inputs=["X_iris", "params:kmeans"],
                outputs=["example_cluster_iris", "example_cluster_model_iris"],
                name="kmeans",
                tags="clustering",
            ),
            node(
                func=PCA,
                inputs=["X_iris", "params:pca"],
                outputs=["X_iris_2d", "pca_model"],
                name="pca",
                tags="dimensionality_reduction",
            ),
            node(
                func=report_clustering_results,
                inputs=["X_iris_2d", "example_cluster_iris", "target", "params:report"],
                outputs="example_cluster_plot_iris",
                name="plot_cluster_results",
                tags="report",
            ),
        ],
        namespace="example_clustering",
    )
