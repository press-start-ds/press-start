from press_start.pipelines.nlp_visualization.nodes import (
    compute_umap_projection,
    load_example_dataset,
    vectorize_nlp,
)
import pandas as pd
import numpy as np


def test_load_example_dataset():
    df_docs = load_example_dataset()
    # Assert we have the right data frame
    assert all(df_docs.columns == ["doc", "filenames", "category"])
    assert df_docs.shape == (18846, 3)
    assert df_docs.category.nunique() == 20


def test_vectorize_nlp():
    df_input = pd.DataFrame(
        [
            ["I love Press Start. It is amazing!"],
            ["I want to contribute to the Press Start package"],
        ],
        columns=["test_doc"],
    )
    arr_output = vectorize_nlp(df_input, {"column_corpus": "test_doc"}, {}).todense()
    arr_expected_rounded = np.array(
        [
            [0.45, 0.0, 0.45, 0.45, 0.45, 0.0, 0.32, 0.32, 0.0, 0.0, 0.0],
            [0.0, 0.33, 0.0, 0.0, 0.0, 0.33, 0.24, 0.24, 0.33, 0.67, 0.33],
        ]
    )
    np.testing.assert_array_almost_equal(arr_output, arr_expected_rounded, 2)


def test_compute_umap_projection():
    np.random.seed(123)
    arr_input = np.round(np.random.rand(5, 10))
    df_output = compute_umap_projection(
        arr_input, {"dim_reduction_args": {"n_neighbors": 2, "random_state": 123}}, {}
    )
    assert df_output.shape == (5, 2)
