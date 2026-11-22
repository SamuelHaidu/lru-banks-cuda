"""Spherical geometry primitives shared by the analysis scripts.

Every numerical routine here is `keras.ops` end to end: no Python-level loop
over vocabulary or embedding-dimension axes, and no numpy in the math path.
Iteration (spherical k-means, the Karcher mean) uses `ops.while_loop`, which
requires every loop variable to keep a fixed shape/dtype across iterations --
that is what shapes the somewhat unusual "carry a running iteration count and
a boolean 'still moving' flag" style used below.

Conventions:
  * All points are rows of a `(N, D)` matrix, already L2-normalized (unit
    vectors on `S^(D-1)`).
  * Similarity between unit vectors is their dot product (cosine similarity).
"""

from typing import Any, Tuple

from model.device import setup_backend

setup_backend()

from keras import ops, random  # noqa: E402

_EPS = 1e-8


def top_k(x: Any, k: int):
    """`ops.top_k` over the full width of a large 1-D vector.

    Some XPU drivers cap the work-group size below what `torch.topk` needs
    for a full-width reduction over ~10k elements (`RuntimeError: ... cannot
    exceed {512, 512, 512}`), independent of tensor size otherwise. Routed
    through plain CPU torch for this one reduction -- the same category of
    device fallback Keras itself already performs transparently for
    `ops.svd`/`ops.eigh` on this hardware, just not automated for `top_k`.
    """
    import torch

    x_cpu = ops.convert_to_numpy(x)
    result = torch.topk(torch.from_numpy(x_cpu), k)
    return ops.convert_to_tensor(result.values.numpy()), ops.convert_to_tensor(result.indices.numpy())


def normalize(x: Any) -> Any:
    """L2-normalizes rows of `x` onto the unit sphere."""
    return ops.normalize(x, axis=-1, order=2)


# ---------------------------------------------------------------------------
# Karcher / Fréchet mean and the log/exp maps used to build a tangent chart.
# ---------------------------------------------------------------------------


def log_map(base: Any, points: Any) -> Any:
    """Riemannian log map from `base` (D,) to each row of `points` (N, D).

    Returns tangent vectors at `base` such that walking along the geodesic in
    that direction by its own length lands back on `points`. Points that
    coincide with `base` map to the zero vector.
    """
    base = ops.reshape(base, (1, -1))
    cos_theta = ops.clip(ops.sum(points * base, axis=-1, keepdims=True), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = ops.arccos(cos_theta)
    proj = points - cos_theta * base
    proj_norm = ops.sqrt(ops.sum(proj * proj, axis=-1, keepdims=True)) + _EPS
    direction = proj / proj_norm
    return direction * theta


def exp_map(base: Any, tangent_vectors: Any) -> Any:
    """Riemannian exponential map: tangent vectors at `base` back onto the sphere."""
    base = ops.reshape(base, (1, -1))
    norm = ops.sqrt(ops.sum(tangent_vectors * tangent_vectors, axis=-1, keepdims=True)) + _EPS
    direction = tangent_vectors / norm
    return ops.cos(norm) * base + ops.sin(norm) * direction


def karcher_mean(points: Any, max_iterations: int = 50, tol: float = 1e-7) -> Any:
    """Intrinsic (geodesic) mean of `points` on the sphere via gradient descent.

    Standard fixed-point iteration: at the current estimate, log-map every
    point into the tangent space, average the tangent vectors, and exp-map the
    average back onto the sphere. Converges to the Fréchet mean for points
    within a hemisphere, which holds here since embeddings are learned and
    concentrate directionally rather than spreading over the full sphere.
    """
    init = normalize(ops.mean(points, axis=0, keepdims=True))[0]

    def cond(mean, step, shift):
        return ops.logical_and(step < max_iterations, shift > tol)

    def body(mean, step, shift):
        tangent = log_map(mean, points)
        avg_tangent = ops.mean(tangent, axis=0)
        new_mean = exp_map(mean, ops.reshape(avg_tangent, (1, -1)))[0]
        new_mean = normalize(ops.reshape(new_mean, (1, -1)))[0]
        new_shift = ops.sqrt(ops.sum((new_mean - mean) ** 2))
        return new_mean, step + 1, new_shift

    mean, _, _ = ops.while_loop(
        cond,
        body,
        (init, ops.convert_to_tensor(0), ops.convert_to_tensor(1.0 + tol)),
        maximum_iterations=max_iterations,
    )
    return mean


# ---------------------------------------------------------------------------
# Principal Geodesic Analysis (PGA): tangent-space PCA at the Karcher mean.
# ---------------------------------------------------------------------------


def pga(points: Any, num_components: int, max_mean_iterations: int = 50) -> Tuple[Any, Any, Any, Any]:
    """Principal Geodesic Analysis of `points` living on a sphere.

    Returns:
        mean:  `(D,)` Karcher mean, the chart's base point.
        basis: `(num_components, D)` orthonormal tangent directions, the
            principal geodesics, ordered by decreasing geodesic variance.
        variance_ratio: `(num_components,)` fraction of total tangent
            variance explained by each component.
        coords: `(N, num_components)` coordinates of every point in the
            tangent chart (projection onto `basis`), i.e. the PGA embedding.
    """
    mean = karcher_mean(points, max_iterations=max_mean_iterations)
    tangent = log_map(mean, points)  # (N, D)

    n = ops.cast(ops.shape(tangent)[0], tangent.dtype)
    cov = ops.matmul(ops.transpose(tangent), tangent) / n  # (D, D)

    eigenvalues, eigenvectors = ops.eigh(cov)  # ascending order
    eigenvalues = ops.flip(eigenvalues, axis=0)
    eigenvectors = ops.flip(eigenvectors, axis=1)

    total_variance = ops.sum(ops.maximum(eigenvalues, 0.0)) + _EPS
    variance_ratio = ops.maximum(eigenvalues, 0.0) / total_variance
    variance_ratio = variance_ratio[:num_components]

    basis = ops.transpose(eigenvectors[:, :num_components])  # (num_components, D)
    coords = ops.matmul(tangent, ops.transpose(basis))  # (N, num_components)

    return mean, basis, variance_ratio, coords


# ---------------------------------------------------------------------------
# Orthogonal Procrustes alignment (for comparing two D-dim point clouds up to
# a rotation/reflection).
# ---------------------------------------------------------------------------


def orthogonal_procrustes(source: Any, target: Any) -> Any:
    """Best orthogonal `R` (D, D) minimizing `||source @ R - target||_F`."""
    m = ops.matmul(ops.transpose(source), target)
    u, _, vt = ops.svd(m)
    return ops.matmul(u, vt)


# ---------------------------------------------------------------------------
# Classical MDS on geodesic distances: a distance-preserving 2D projection.
# ---------------------------------------------------------------------------


def geodesic_mds(points: Any, num_components: int = 2) -> Any:
    """Classical (Torgerson) MDS embedding of `points`, using great-circle distance.

    Unlike PGA (a linear tangent-plane projection, accurate only near the
    chart's base point), this directly minimizes distortion of the full
    pairwise geodesic-distance matrix: double-center the squared geodesic
    distances and take the leading eigenvectors, scaled by `sqrt(eigenvalue)`
    -- the classical result that this recovers the best rank-`num_components`
    Euclidean approximation to the geodesic metric (in a least-squares sense).

    Quadratic in the number of points (builds an `(N, N)` distance matrix), so
    intended for one-off visualization, not as an inner-loop routine.
    """
    n = ops.shape(points)[0]
    cos_sim = ops.clip(ops.matmul(points, ops.transpose(points)), -1.0 + 1e-7, 1.0 - 1e-7)
    geodesic = ops.arccos(cos_sim)
    squared = geodesic * geodesic

    n_f = ops.cast(n, squared.dtype)
    row_mean = ops.mean(squared, axis=1, keepdims=True)
    col_mean = ops.mean(squared, axis=0, keepdims=True)
    grand_mean = ops.mean(squared)
    gram = -0.5 * (squared - row_mean - col_mean + grand_mean)

    eigenvalues, eigenvectors = ops.eigh(gram)  # ascending order
    eigenvalues = ops.flip(eigenvalues, axis=0)[:num_components]
    eigenvectors = ops.flip(eigenvectors, axis=1)[:, :num_components]

    scale = ops.sqrt(ops.maximum(eigenvalues, 0.0))
    return eigenvectors * ops.reshape(scale, (1, -1))


# ---------------------------------------------------------------------------
# Spherical k-means.
# ---------------------------------------------------------------------------


def _kmeans_plus_plus_init(points: Any, k: int, seed: Any) -> Any:
    """k-means++ seeding under cosine (spherical) distance, keras.ops only."""
    n = ops.shape(points)[0]
    seed_gen = random.SeedGenerator(seed)

    first_idx = random.randint((1,), minval=0, maxval=n, seed=seed_gen)
    centroid_idx = ops.reshape(first_idx, (1,))

    def cond(centroid_idx, step):
        return step < k

    def body(centroid_idx, step):
        centroids = ops.take(points, centroid_idx, axis=0)
        sims = ops.matmul(points, ops.transpose(centroids))  # (N, step)
        best_sim = ops.max(sims, axis=-1)
        dist_sq = ops.maximum(1.0 - best_sim, 0.0) ** 2 + _EPS
        probs = dist_sq / ops.sum(dist_sq)
        next_idx = random.categorical(ops.log(ops.reshape(probs, (1, -1))), 1, seed=seed_gen)
        next_idx = ops.cast(ops.reshape(next_idx, (1,)), centroid_idx.dtype)
        return ops.concatenate([centroid_idx, next_idx], axis=0), step + 1

    centroid_idx, _ = ops.while_loop(
        cond,
        body,
        (centroid_idx, ops.convert_to_tensor(1)),
        maximum_iterations=k - 1,
    )
    return ops.take(points, centroid_idx, axis=0)


def spherical_kmeans(
    points: Any,
    k: int,
    max_iterations: int = 100,
    seed: int = 0,
) -> Tuple[Any, Any, Any]:
    """Spherical k-means: cosine-similarity assignment, normalized-mean update.

    Returns:
        centroids: `(k, D)` unit-norm cluster centroids.
        assignments: `(N,)` int32 cluster id per point.
        num_iterations: scalar int, iterations actually run before convergence.
    """
    centroids0 = _kmeans_plus_plus_init(points, k, seed)
    n = ops.shape(points)[0]
    assignments0 = ops.zeros((n,), dtype="int32") - 1

    def cond(centroids, assignments, step, num_changed):
        return ops.logical_and(step < max_iterations, num_changed > 0)

    def body(centroids, assignments, step, num_changed):
        sims = ops.matmul(points, ops.transpose(centroids))  # (N, k)
        new_assignments = ops.cast(ops.argmax(sims, axis=-1), "int32")

        one_hot = ops.cast(ops.equal(ops.reshape(new_assignments, (-1, 1)), ops.arange(k, dtype="int32")), points.dtype)
        counts = ops.sum(one_hot, axis=0, keepdims=True)  # (1, k)
        sums = ops.matmul(ops.transpose(one_hot), points)  # (k, D)
        safe_counts = ops.transpose(ops.maximum(counts, 1.0))
        means = sums / safe_counts
        empty = ops.reshape(ops.equal(ops.transpose(counts)[:, 0], 0.0), (k, 1))
        new_centroids = ops.where(empty, centroids, normalize(means))

        changed = ops.sum(ops.cast(ops.not_equal(new_assignments, assignments), "int32"))
        return new_centroids, new_assignments, step + 1, changed

    centroids, assignments, num_iterations, _ = ops.while_loop(
        cond,
        body,
        (centroids0, assignments0, ops.convert_to_tensor(0), ops.convert_to_tensor(n)),
        maximum_iterations=max_iterations,
    )
    return centroids, assignments, num_iterations


# ---------------------------------------------------------------------------
# Cluster-agreement metrics (contingency-table based, keras.ops only).
# ---------------------------------------------------------------------------


def nearest_neighbors(points: Any, num_neighbors: int) -> Any:
    """`(N, num_neighbors)` int ids of each point's nearest neighbors by cosine similarity.

    Excludes the point itself (`num_neighbors` genuine neighbors, not
    `num_neighbors - 1`). Uses the batched `top_k` fallback since this is a
    row-wise reduction over the full vocabulary width.
    """
    sims = ops.matmul(points, ops.transpose(points))
    n = ops.shape(points)[0]
    self_mask = ops.eye(n, dtype="bool")
    sims = ops.where(self_mask, ops.full_like(sims, -2.0), sims)
    _, indices = top_k(sims, k=num_neighbors)
    return ops.cast(indices, "int32")


def neighbor_overlap(neighbors_a: Any, neighbors_b: Any, num_ids: int) -> Any:
    """Per-point fraction of `neighbors_a` rows also present in `neighbors_b`.

    `neighbors_a`/`neighbors_b`: `(N, k)` neighbor-id matrices over the same
    `N` points (e.g. from `nearest_neighbors` on two different tables).
    Returns `(N,)` overlap fractions in `[0, 1]`.
    """
    n, k = ops.shape(neighbors_a)[0], ops.shape(neighbors_a)[1]
    one_hot_a = ops.cast(ops.equal(ops.reshape(neighbors_a, (n, k, 1)), ops.arange(num_ids, dtype="int32")), "float32")
    one_hot_b = ops.cast(ops.equal(ops.reshape(neighbors_b, (n, k, 1)), ops.arange(num_ids, dtype="int32")), "float32")
    set_a = ops.sum(one_hot_a, axis=1)  # (N, num_ids), multi-hot membership
    set_b = ops.sum(one_hot_b, axis=1)
    overlap_counts = ops.sum(set_a * set_b, axis=-1)
    return overlap_counts / ops.cast(k, overlap_counts.dtype)


def pairwise_similarity_correlation(points_a: Any, points_b: Any, sample_indices: Any) -> Any:
    """Pearson correlation between two tables' cosine similarities, sampled pairs.

    `sample_indices`: `(M, 2)` int array of row-index pairs into both tables
    (same `N` points, two different embeddings). Comparing the *full*
    `N x N` similarity matrices is unnecessary for a single correlation
    coefficient -- a random sample of pairs gives the same statistic to
    within sampling noise, without needing a second `(N, N)` matrix.
    """
    idx_i = sample_indices[:, 0]
    idx_j = sample_indices[:, 1]
    a_i = ops.take(points_a, idx_i, axis=0)
    a_j = ops.take(points_a, idx_j, axis=0)
    b_i = ops.take(points_b, idx_i, axis=0)
    b_j = ops.take(points_b, idx_j, axis=0)

    sim_a = ops.sum(a_i * a_j, axis=-1)
    sim_b = ops.sum(b_i * b_j, axis=-1)

    centered_a = sim_a - ops.mean(sim_a)
    centered_b = sim_b - ops.mean(sim_b)
    numerator = ops.sum(centered_a * centered_b)
    denominator = ops.sqrt(ops.sum(centered_a * centered_a) * ops.sum(centered_b * centered_b)) + _EPS
    correlation = numerator / denominator
    return correlation, sim_a, sim_b


def group_discriminability(points: Any, group_ids: Any) -> Any:
    """How well a small group of points is spread out (vs. bunched together).

    `group_ids`: `(g,)` int ids of a token's own neighborhood (e.g. a token
    plus its nearest neighbors). Returns `(mean_off_diag_sim, participation_ratio)`:
      * `mean_off_diag_sim`: average pairwise cosine similarity within the
        group, excluding self-pairs -- higher means the group is bunched
        closer together (less discriminable).
      * `participation_ratio`: `(sum(eigenvalues))^2 / sum(eigenvalues^2)` of
        the group's covariance, normalized to `[0, 1]` by the ambient
        dimension -- an effective-rank measure of how many independent
        directions the group's variance actually spreads across (1.0 = spread
        evenly across every available axis, near 0 = collapsed onto one axis).
    """
    group = ops.take(points, group_ids, axis=0)
    g = ops.shape(group)[0]
    dim = ops.shape(group)[1]

    sims = ops.matmul(group, ops.transpose(group))
    off_diag_sum = ops.sum(sims) - ops.sum(ops.diagonal(sims))
    denom = ops.cast(g * (g - 1), sims.dtype)
    mean_off_diag_sim = off_diag_sum / (denom + _EPS)

    centered = group - ops.mean(group, axis=0, keepdims=True)
    cov = ops.matmul(ops.transpose(centered), centered)
    eigenvalues = ops.eigh(cov)[0]
    eigenvalues = ops.maximum(eigenvalues, 0.0)
    participation = (ops.sum(eigenvalues) ** 2) / (ops.sum(eigenvalues * eigenvalues) + _EPS)
    max_rank = ops.cast(ops.minimum(g - 1, dim), participation.dtype)
    participation_ratio = participation / (max_rank + _EPS)

    return mean_off_diag_sim, participation_ratio


def contingency_matrix(labels_a: Any, labels_b: Any, k_a: int, k_b: int) -> Any:
    """`(k_a, k_b)` co-occurrence counts between two clusterings of the same points."""
    one_hot_a = ops.cast(ops.equal(ops.reshape(labels_a, (-1, 1)), ops.arange(k_a, dtype="int32")), "float32")
    one_hot_b = ops.cast(ops.equal(ops.reshape(labels_b, (-1, 1)), ops.arange(k_b, dtype="int32")), "float32")
    return ops.matmul(ops.transpose(one_hot_a), one_hot_b)


def _entropy(counts: Any, total: Any) -> Any:
    p = counts / total
    p = ops.where(p > 0, p, ops.ones_like(p))
    return -ops.sum(ops.where(counts > 0, p * ops.log(p), ops.zeros_like(p)))


def adjusted_rand_index(contingency: Any) -> Any:
    """ARI from a contingency matrix, `keras.ops` only (no scipy/sklearn)."""
    n = ops.sum(contingency)

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_comb_c = ops.sum(comb2(contingency))
    row_sums = ops.sum(contingency, axis=1)
    col_sums = ops.sum(contingency, axis=0)
    sum_comb_rows = ops.sum(comb2(row_sums))
    sum_comb_cols = ops.sum(comb2(col_sums))
    comb_n = comb2(n)

    expected_index = sum_comb_rows * sum_comb_cols / (comb_n + _EPS)
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = max_index - expected_index
    ari = (sum_comb_c - expected_index) / (denom + _EPS)
    return ari


def normalized_mutual_info(contingency: Any) -> Any:
    """NMI (arithmetic mean normalization) from a contingency matrix."""
    n = ops.sum(contingency)
    row_sums = ops.sum(contingency, axis=1, keepdims=True)
    col_sums = ops.sum(contingency, axis=0, keepdims=True)

    pij = contingency / n
    pi = row_sums / n
    pj = col_sums / n
    outer = pi * pj

    ratio = ops.where(contingency > 0, pij / (outer + _EPS), ops.ones_like(pij))
    mi = ops.sum(ops.where(contingency > 0, pij * ops.log(ratio + _EPS), ops.zeros_like(pij)))

    h_a = _entropy(row_sums[:, 0], n)
    h_b = _entropy(col_sums[0, :], n)
    denom = 0.5 * (h_a + h_b)
    return mi / (denom + _EPS)
