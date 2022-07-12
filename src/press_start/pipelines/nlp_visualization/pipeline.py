from kedro.pipeline import Pipeline, node

from .nodes import (
    load_example_dataset,
    generate_viz,
    vectorize_nlp,
    compute_umap_projection,
    cluster_data,
)


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
            node(
                func=vectorize_nlp,
                inputs=dict(
                    df_data="nlp_example_dataset",
                    params_dict="params:nlp_visualization",
                    general_params_dict="params:general",
                ),
                outputs="nlp_vectorized",
                name="vectorize_nlp",
                tags="nlp_visualization",
            ),
            node(
                func=compute_umap_projection,
                inputs=dict(
                    arr_data="nlp_vectorized",
                    params_dict="params:nlp_visualization",
                    general_params_dict="params:general",
                ),
                outputs="nlp_projection",
                name="compute_umap_projection",
                tags="nlp_visualization",
            ),
            node(
                func=cluster_data,
                inputs=dict(
                    arr_data="nlp_vectorized",
                    params_dict="params:nlp_visualization",
                    general_params_dict="params:general",
                ),
                outputs="nlp_cluster_labels",
                name="cluster_data",
                tags="nlp_visualization",
            ),
            node(
                func=generate_viz,
                inputs=dict(
                    df_data="nlp_example_dataset",
                    df_projection="nlp_projection",
                    cluster_labels="nlp_cluster_labels",
                    params_dict="params:nlp_visualization",
                    general_params_dict="params:general",
                ),
                outputs="nlp_visualization",
                name="generate_visualization",
                tags="nlp_visualization",
            ),
        ]
    )
