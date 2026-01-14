#!/usr/bin/env Rscript

library(argparse)
library(jsonlite)
library(mclust)

parser <- ArgumentParser(description = "Benchmarking entrypoint")

parser$add_argument(
  "--output_dir", "-o",
  dest = "output_dir", type = "character",
  help = "output directory where files will be saved",
  default = getwd(), required = TRUE
)
parser$add_argument(
  "--name", "-n",
  dest = "name", type = "character",
  help = "name of the module",
  required = TRUE
)
parser$add_argument(
  "--timings.json",
  dest = "timings_path", type = "character",
  help = "timings json path", required = TRUE
)
parser$add_argument(
  "--cluster.tsv",
  dest = "cluster_path", type = "character",
  help = "cluster tsv path", required = TRUE
)

args <- parser$parse_args()

clusters <- read.csv(args$cluster_path, sep = "\t")
metrics_path <- file.path(args$output_dir, paste0(args$name, ".metrics.json"))

metrics <- list(
  ari = list(leiden = NA_real_, louvain = NA_real_),
  timings = NULL
)

metrics$timings <- fromJSON(args$timings_path)

metrics$ari$leiden <- adjustedRandIndex(metrics$truths, metrics$leiden)
metrics$ari$louvain <- adjustedRandIndex(metrics$truths, metrics$louvain)

write_json(
  metrics, metrics_path,
  auto_unbox = TRUE, pretty = TRUE
)
