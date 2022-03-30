"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from press_start.pipelines.feature_analysis import pipeline as fa
from press_start.pipelines.feature_selection import pipeline as fs


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    feature_analysis_pipeline = fa.create_pipeline()
    feature_selection_pipeline = fs.create_pipeline()

    return {
        "__default__": feature_analysis_pipeline + feature_selection_pipeline,
    }
