"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from press_start.pipelines.feature_analysis import pipeline as fa
from kedro.framework.session import get_current_session


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    session = get_current_session()
    context = session.load_context()
    params = {
        param_name[7:]: context.catalog.load(param_name)
        for param_name in context.catalog.list() if
        param_name.startswith('params:')
    }

    feature_analysis_pipeline = fa.create_pipeline(params)

    return {
        "__default__": Pipeline([feature_analysis_pipeline])
    }
