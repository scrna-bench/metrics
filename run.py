#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

METRIC_MAP = {
    "agreement": {
        "ari": adjusted_rand_score,
        "nmi": normalized_mutual_info_score,
        "fowlkes_mallows": fowlkes_mallows_score,
        "homogeneity": homogeneity_score,
        "completeness": completeness_score,
        "v_measure": v_measure_score,
    },
    "structure": {
        "silhouette": silhouette_score,
        "davies_bouldin": davies_bouldin_score,
        "calinski_harabasz": calinski_harabasz_score,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Benchmarking entrypoint")
    parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        help="output directory where files will be saved",
        default=".",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--name",
        "-n",
        dest="name",
        help="name of the module",
        required=True,
    )
    parser.add_argument(
        "--timings.json",
        dest="timings_path",
        help="timings json path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cluster.tsv",
        dest="cluster_path",
        help="cluster tsv path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pca.tsv",
        dest="pca_path",
        help="pca tsv path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data.clusters_truth",
        dest="clusters_truth_path",
        help="clusters_truth tsv path",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    metrics_path = args.output_dir / f"{args.name}.metrics.json"

    clusters = pd.read_csv(args.cluster_path, sep="\t")
    pcas = pd.read_csv(args.pca_path, sep="\t")
    truths = pd.read_csv(args.clusters_truth_path, sep="\t").dropna(subset=["truths"])
    merged = clusters.merge(truths, on="cell_id", how="inner")

    # number of cells dropped due to no truth label
    n_dropped_rows = len(clusters) - len(merged)

    truth_labels = merged["truths"]
    leiden_labels = merged["leiden"]
    louvain_labels = merged["louvain"]

    merged_pca = clusters.merge(pcas, on="cell_id", how="inner")
    pca_matrix = merged_pca[[c for c in merged_pca.columns if c.startswith("PC")]]

    with args.timings_path.open() as handle:
        timings = json.load(handle)

    metrics = {
        "agreement": {
            key: {
                "leiden": fn(truth_labels, leiden_labels),
                "louvain": fn(truth_labels, louvain_labels),
            }
            for key, fn in METRIC_MAP["agreement"].items()
        },
        "structure": {
            key: {
                "leiden": fn(pca_matrix, merged_pca["leiden"]),
                "louvain": fn(pca_matrix, merged_pca["louvain"]),
            }
            for key, fn in METRIC_MAP["structure"].items()
        },
        "n_clusters": {
            "leiden": clusters["leiden"].nunique(),
            "louvain": clusters["louvain"].nunique(),
        },
        "dropped_cells": n_dropped_rows,
        "timings": timings,
    }

    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=False)


if __name__ == "__main__":
    main()
