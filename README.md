# Metrics

Metrics computed for the scRNA pipelines benchmark.

## Agreement metrics

Reported agreement of Leiden and Louvain predicted clusters with true labels:

- Adjusted Rand Index
- Normalized Mutual Information
- Fowlkes-Mallows
- Homogeneity
- Completeness
- V-measure

## Structure metrics

Computed on PCA coordinates for predicted clusters of Leiden and Louvain:

- Silhouette score
- Davies-Bouldin score
- Calinski-Harabasz score

## Other outputs

- `n_clusters`: number of clusters for Leiden and Louvain
- `dropped_cells`: number of cells removed from agreement metrics computations due to missing truth labels
- `timings`: runtime breakdown from the pipeline, with `louvain` and `leiden`
  stored as average seconds per clustering invocation during resolution search
- `clustering_info`: clustering search metadata with `resolutions` and `num_runs`
