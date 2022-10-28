"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from press_start.pipelines.feature_analysis import pipeline as fa
from press_start.pipelines.feature_selection import pipeline as fs
from press_start.pipelines.data_split import pipeline as ds
from press_start.pipelines.nlp_visualization import pipeline as nlp_viz
from press_start.pipelines.clustering import pipeline as kmeans


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    feature_analysis_pipeline = fa.create_pipeline()
    feature_selection_pipeline = fs.create_pipeline()
    data_split_pipeline = ds.create_pipeline()
    nlp_visualization_pipeline = nlp_viz.create_pipeline()
    kmeans_pipeline = kmeans.create_pipeline()

    return {
        "__default__": (
            data_split_pipeline
            + feature_analysis_pipeline
            + feature_selection_pipeline
            + nlp_visualization_pipeline
            + kmeans_pipeline
        ),
        "nlp_visualization": nlp_visualization_pipeline,
        "example_clustering_kmeans": kmeans_pipeline,
    }
