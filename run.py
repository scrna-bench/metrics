#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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
        "--data.clusters_truth",
        dest="clusters_truth_path",
        help="clusters_truth tsv path",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    metrics_path = args.output_dir / f"{args.name}.metrics.json"

    clusters = pd.read_csv(args.cluster_path, sep="\t")
    truths = pd.read_csv(args.clusters_truth_path, sep="\t").dropna(subset=["truths"])
    merged = clusters.merge(truths, on="cell_id", how="inner")

    # number of cells dropped due to no truth label
    n_dropped_rows = len(clusters) - len(merged)

    truth_labels = merged["truths"]
    leiden_labels = merged["leiden"]
    louvain_labels = merged["louvain"]

    with args.timings_path.open() as handle:
        timings = json.load(handle)

    metrics = {
        "ari": {
            "leiden": adjusted_rand_score(truth_labels, leiden_labels),
            "louvain": adjusted_rand_score(truth_labels, louvain_labels),
        },
        "nmi": {
            "leiden": normalized_mutual_info_score(truth_labels, leiden_labels),
            "louvain": normalized_mutual_info_score(truth_labels, louvain_labels),
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
