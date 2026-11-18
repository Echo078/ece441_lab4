"""
Lab 4: cluster-based nonparametric statistics (Maris & Oostenveld, 2007).

Function 1, `nonparametric_test`, is the single-sensor cluster-mass statistic of
section 3.1.1 of the paper; function 2, `permutation_test`, is the permutation
(Monte Carlo) procedure of section 2.3 that turns it into a p-value.

Both work channel by channel on arrays shaped (n_epochs, n_channels, n_times).
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage, stats

SAMPLE_ALPHA = 0.05  # threshold that selects samples for clustering (paper: 97.5th t quantile)
DEFAULT_N_PERM = 1000

Cluster = Optional[Tuple[int, int]]  # (first sample, last sample + 1) of a channel's biggest cluster


def _check_datasets(dataset_a: np.ndarray, dataset_b: np.ndarray) -> None:
    if dataset_a.ndim != 3 or dataset_b.ndim != 3:
        raise ValueError("datasets must be shaped (n_epochs, n_channels, n_times)")
    if dataset_a.shape[1:] != dataset_b.shape[1:]:
        raise ValueError(
            f"datasets differ in (n_channels, n_times): {dataset_a.shape[1:]} vs {dataset_b.shape[1:]}"
        )
    if len(dataset_a) < 2 or len(dataset_b) < 2:
        raise ValueError("each condition needs at least 2 epochs for a t-test")


def _largest_cluster(t_values: np.ndarray, significant: np.ndarray) -> Tuple[float, Cluster]:
    """Largest (in absolute value) sum of t over runs of adjacent significant samples.

    Positive and negative t-values are clustered separately, as the paper
    prescribes for a two-sided test, so a cluster never mixes signs.
    """
    best_stat, best_cluster = 0.0, None
    for same_sign in (significant & (t_values > 0), significant & (t_values < 0)):
        labels, n_clusters = ndimage.label(same_sign)
        for slc in ndimage.find_objects(labels):
            if slc is None:
                continue
            cluster_stat = float(t_values[slc[0]].sum())
            if abs(cluster_stat) > abs(best_stat):
                best_stat, best_cluster = cluster_stat, (slc[0].start, slc[0].stop)
    return best_stat, best_cluster


def nonparametric_test(
    dataset_a: np.ndarray,
    dataset_b: np.ndarray,
    return_clusters: bool = False,
):
    """Function 1: cluster-level test statistic of each channel (section 3.1.1).

    1. Two-sample t-test at every (channel, time) sample across the two conditions
       -> t-statistics and p-values, both shaped (n_channels, n_times).
    2. For each channel: find clusters of adjacent samples with p < 0.05 (positive
       and negative t separately), sum the t-values in each cluster and keep the
       cluster whose sum is largest in absolute value.

    Parameters
    ----------
    dataset_a, dataset_b : ndarray, shape (n_epochs, n_channels, n_times)
        Voltage traces of condition A and B (the epoch counts may differ).
    return_clusters : bool
        Also return where each channel's biggest cluster is.

    Returns
    -------
    clus_tstat : ndarray, shape (n_channels,)
        Signed sum of t of each channel's biggest cluster (0 if no sample has p < 0.05).
    clusters : list of (start, stop) or None, only if return_clusters
        Sample range [start, stop) of that cluster.
    """
    _check_datasets(dataset_a, dataset_b)
    t_values, p_values = stats.ttest_ind(dataset_a, dataset_b, axis=0)
    significant = p_values < SAMPLE_ALPHA

    clus_tstat = np.zeros(t_values.shape[0])
    clusters: List[Cluster] = []
    for channel in range(t_values.shape[0]):
        clus_tstat[channel], cluster = _largest_cluster(t_values[channel], significant[channel])
        clusters.append(cluster)

    if return_clusters:
        return clus_tstat, clusters
    return clus_tstat


def permutation_test(
    dataset_a: np.ndarray,
    dataset_b: np.ndarray,
    n_perm: int = DEFAULT_N_PERM,
    seed: Optional[int] = 0,
    return_null: bool = False,
):
    """Function 2: permutation p-value of each channel's biggest cluster (section 2.3).

    1. Function 1 on the real data gives the observed statistics.
    2. Pool the epochs of both conditions (n_a + n_b, n_channels, n_times).
    3. Shuffle the pooled epochs and split them into two sets of the original sizes.
    4. Function 1 on that random partition gives one draw of the null distribution.
    5. Repeat 3-4 n_perm times.
    6. p = proportion of random partitions whose statistic is larger (in absolute
       value, i.e. a two-sided test) than the observed one.

    Parameters
    ----------
    dataset_a, dataset_b : ndarray, shape (n_epochs, n_channels, n_times)
    n_perm : int, optional
        Number of random partitions, by default 1000.
    seed : int or None
        Seed of the random generator (None = not reproducible).
    return_null : bool
        Also return the null distribution, shape (n_perm, n_channels).

    Returns
    -------
    clus_pvals : ndarray, shape (n_channels,)
    null : ndarray, only if return_null
    """
    _check_datasets(dataset_a, dataset_b)
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")

    observed = nonparametric_test(dataset_a, dataset_b)
    pooled = np.concatenate([dataset_a, dataset_b], axis=0)
    n_a = len(dataset_a)
    rng = np.random.default_rng(seed)

    null = np.empty((n_perm, pooled.shape[1]))
    for i in range(n_perm):
        order = rng.permutation(len(pooled))
        null[i] = nonparametric_test(pooled[order[:n_a]], pooled[order[n_a:]])

    # ">=" so that a channel without any cluster (statistic 0) gets p = 1, not 0
    clus_pvals = (np.abs(null) >= np.abs(observed)).mean(axis=0)
    if return_null:
        return clus_pvals, null
    return clus_pvals
