from typing import Dict, Union
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tempfile
import datapane as dp
import plotly.express as px
import umap
import os
from sklearn.datasets import fetch_20newsgroups
from press_start.params import ParamsNLPViz

log = logging.getLogger(__name__)


def load_example_dataset():
    newsgroups = fetch_20newsgroups(subset="all")

    keys = ("data", "filenames", "target")
    target_names = np.array(newsgroups["target_names"])
    return (
        pd.DataFrame([newsgroups[key] for key in keys], index=keys)
        .T.assign(target=lambda df: df.target.apply(lambda idx: target_names[idx]))
        .rename(columns={"data": "doc", "target": "category"})
    )


def vectorize_nlp(
    df_data: pd.DataFrame,
    params_dict: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> np.ndarray:
    params = ParamsNLPViz(params_dict)
    log.debug(f"Pipeline params loaded: {params_dict}")
    log.info("Computing TF-IDF")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_data[params.column_corpus])
    log.info(f"TF-IDF computed, resulting in {tfidf_matrix.shape[1]} columns")
    return tfidf_matrix


def compute_umap_projection(
    arr_data: np.ndarray,
    params_dict: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> pd.DataFrame:
    params = ParamsNLPViz(params_dict)
    log.debug(f"Pipeline params loaded: {params_dict}")
    log.info("Computing UMAP")
    reducer = umap.UMAP(**params.dim_reduction_args)
    umap_embeds = reducer.fit_transform(arr_data)
    log.info("UMAP computed")
    return pd.DataFrame(umap_embeds, columns=("x", "y"))


def cluster_data(
    arr_data: np.ndarray,
    params_dict: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> np.ndarray:
    params = ParamsNLPViz(params_dict)
    log.debug(f"Pipeline params loaded: {params_dict}")
    if params.data_clustering_method:
        log.info("Clustering data")
        clustering = KMeans().fit(arr_data)
        return clustering.labels_
    return np.array([])


def generate_viz(
    df_data: pd.DataFrame,
    df_projection: pd.DataFrame,
    cluster_labels: np.ndarray,
    params_dict: Dict[str, Union[int, float]],
    general_params_dict: Dict[str, Dict],
) -> str:
    params = ParamsNLPViz(params_dict)
    log.debug(f"Pipeline params loaded: {params_dict}")
    df_doc = df_data[params.column_corpus].str[:100]
    if params.column_category in df_data.columns:
        labels = df_data[params.column_category]
    else:
        labels = [str(i) for i in cluster_labels]
    fig = px.scatter(
        df_projection.assign(labels=labels, summary=df_doc),
        x="x",
        y="y",
        color="labels",
        hover_name="summary",
    )
    log.debug("Plotly plot generated")
    # return fig.to_html(include_plotlyjs="cdn")

    # TODO: Use `to_string` method from datapane instead of files
    with tempfile.TemporaryDirectory() as folder:
        path_file = os.path.join(folder, "report.html")
        dp.Report(
            dp.Page(
                title="2D visualization",
                blocks=[
                    "### UMAP visualization",
                    dp.Plot(fig),
                    "### Sample snippets",
                    dp.DataTable(
                        df_doc.to_frame()
                        .assign(cluster=labels)
                        .groupby("cluster", group_keys=False)
                        .apply(lambda df: df.sample(min(10, round(df.shape[0] * 0.1))))
                    ),
                ],
            ),
        ).save(
            path=path_file, formatting=dp.ReportFormatting(width=dp.ReportWidth.NARROW)
        )
        with open(path_file, "r") as f:
            html_content = f.read()

        return html_content
