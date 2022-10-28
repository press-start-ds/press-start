import plotly.express as px
import pandas as pd


def report_clustering_results(df_features, df_cluster, df_target, params):
    return px.scatter(
        pd.concat(
            (
                df_features.set_axis(["x1", "x2"], axis=1),
                df_cluster.set_axis(["cluster"], axis=1)["cluster"].astype(str),
                df_target.set_axis(["target"], axis=1)["target"].astype(str),
            ),
            axis=1,
        ),
        x="x1",
        y="x2",
        color="cluster",
        symbol="target",
    ).to_html()
