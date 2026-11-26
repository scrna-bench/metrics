#!/usr/bin/env python3
"""METRIC gather: score every PIPELINE run of one dataset.

ob (api 0.8) calls this once per DATA node with
  --from-files <lineage.json>   the dataset's pipeline runs (gather manifest)
  --data_clusters_truth <tsv>   that dataset's truth labels (stage input)
  --output metrics_tsv=<path> --output timings_tsv=<path>
and it writes one metrics row per run and clustering algorithm, and one
timings row per recorded phase, each with the run's lineage columns."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import polars as pl  # noqa: E402

from events import phases  # noqa: E402
from fanin import frame, manifest_members, member_lineage_cols  # noqa: E402
from scores import COLUMNS, TYPES, read_truths, score_run  # noqa: E402

INPUTS = ("methods_clusters", "methods_pca", "methods_clustering_info", "methods_events")
TIMING_COLUMNS = ["phase", "seconds", "num_runs", "per_run_seconds", "resolution", "start", "end", "host"]
TIMING_TYPES = {"seconds": pl.Float64, "num_runs": pl.Int64, "per_run_seconds": pl.Float64,
                "resolution": pl.Float64}


def _key_path(s):
    key, sep, path = s.partition("=")
    if not sep or not key or not path:
        raise argparse.ArgumentTypeError(f"--output wants ID=PATH, got {s!r}")
    return key, path


def score_member(member, truths_path):
    truths = read_truths(truths_path)
    lineage = member_lineage_cols(member, "pipeline")
    metric_rows = [{**lineage, **r} for r in score_run(member["files"], truths)]
    timing_rows = []
    for p in phases(member["files"]["methods_events"]):
        runs = p.get("num_runs")
        p["per_run_seconds"] = p["seconds"] / runs if runs and p["seconds"] is not None else None
        timing_rows.append({**lineage, **p})
    return metric_rows, timing_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--output_dir")
    p.add_argument("--name")
    p.add_argument("--output", type=_key_path, action="append", default=[])
    p.add_argument("--from-files", dest="from_files", required=True, help="gather manifest (lineage.json)")
    p.add_argument("--data_clusters_truth", required=True, help="truth labels of this dataset")
    args = p.parse_args()
    outputs = dict(args.output)
    for oid in ("metrics_tsv", "timings_tsv"):
        if oid not in outputs:
            p.error(f"missing --output {oid}=PATH")

    members = manifest_members(args.from_files, *INPUTS)
    workers = max(1, min(len(members), int(os.environ.get("OMP_NUM_THREADS", "1"))))
    print(f"scoring {len(members)} pipeline runs with {workers} worker(s)", file=sys.stderr)

    metric_rows, timing_rows = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for m, t in pool.map(score_member, members, [args.data_clusters_truth] * len(members)):
            metric_rows += m
            timing_rows += t

    frame(metric_rows, COLUMNS, TYPES).write_csv(outputs["metrics_tsv"], separator="\t")
    frame(timing_rows, TIMING_COLUMNS, TIMING_TYPES).write_csv(outputs["timings_tsv"], separator="\t")


if __name__ == "__main__":
    main()
