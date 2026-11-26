"""Clustering metrics for one pipeline run against the truth labels."""

import json

import numpy as np
import polars as pl
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

AGREEMENT = {
    "ari": adjusted_rand_score,
    "nmi": normalized_mutual_info_score,
    "fowlkes_mallows": fowlkes_mallows_score,
    "homogeneity": homogeneity_score,
    "completeness": completeness_score,
    "v_measure": v_measure_score,
}
STRUCTURE = {
    "silhouette": silhouette_score,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
}
ALGORITHMS = ("louvain", "leiden")
COLUMNS = (
    ["algorithm", "n_clusters", "resolution", "num_runs", "dropped_cells"]
    + list(AGREEMENT) + list(STRUCTURE)
)
TYPES = {"n_clusters": pl.Int64, "resolution": pl.Float64, "num_runs": pl.Int64,
         "dropped_cells": pl.Int64} | {k: pl.Float64 for k in [*AGREEMENT, *STRUCTURE]}

LABELS = {"cell_id": pl.String, "louvain": pl.String, "leiden": pl.String}


def read_truths(path):
    return pl.read_csv(
        path, separator="\t", schema_overrides={"cell_id": pl.String, "truths": pl.String}
    ).drop_nulls("truths")


def _structure(fn, X, labels):
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(fn(X, labels))


def score_run(files, truths):
    """files: the member's output id -> path. truths: cell_id/truths table.
    Returns one dict per clustering algorithm the pipeline produced (a
    pipeline without Louvain leaves that column empty and gets no row)."""
    clusters = pl.read_csv(files["methods_clusters"], separator="\t", schema_overrides=LABELS)
    pca = pl.read_csv(files["methods_pca"], separator="\t", schema_overrides={"cell_id": pl.String})
    with open(files["methods_clustering_info"]) as fh:
        info = json.load(fh)

    merged = clusters.join(truths, on="cell_id", how="inner")
    merged_pca = clusters.join(pca, on="cell_id", how="inner")
    X = merged_pca.select(pl.col(r"^PC\d+$")).to_numpy()

    rows = []
    for alg in ALGORITHMS:
        if clusters[alg].null_count() == clusters.height:
            continue
        row = {
            "algorithm": alg,
            "n_clusters": clusters[alg].n_unique(),
            "resolution": info["resolutions"].get(alg),
            "num_runs": info["num_runs"].get(alg),
            # cells the pipeline kept that have no truth label
            "dropped_cells": clusters.height - merged.height,
        }
        truth, labels = merged["truths"].to_numpy(), merged[alg].to_numpy()
        for name, fn in AGREEMENT.items():
            row[name] = float(fn(truth, labels))
        for name, fn in STRUCTURE.items():
            row[name] = _structure(fn, X, merged_pca[alg].to_numpy())
        rows.append(row)
    return rows
