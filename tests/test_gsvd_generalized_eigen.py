import numpy as np

from easygsvd.gsvd import gsvd


def _random_full_column_rank_pair(N, seed, max_tries=10):
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        M = int(rng.integers(80, 131))
        K = int(rng.integers(80, 131))
        A = rng.standard_normal((M, N))
        L = rng.standard_normal((K, N))
        if np.linalg.matrix_rank(np.vstack([A, L])) == N:
            return A, L
    raise RuntimeError("Failed to generate full-column-rank stacked matrix within max_tries.")


def test_generalized_singular_values_satisfy_normal_eigen_relations():
    N = 100
    tol = 1e-10

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=False)

        if G.X2.shape[1] == 0:
            continue  # no intersection part to test

        ATA = A.T @ A
        LTL = L.T @ L

        for gamma, x in zip(G.gamma_check, G.X2.T):
            lhs = ATA @ x
            rhs = (gamma ** 2) * (LTL @ x)
            denom = max(1.0, np.linalg.norm(lhs, 2), np.linalg.norm(rhs, 2))
            assert np.linalg.norm(lhs - rhs, 2) / denom <= tol


def test_gamma_matches_svd_when_L_is_identity():
    N = 100
    tol = 1e-10

    for seed in range(50):
        rng = np.random.default_rng(seed)
        M = int(rng.integers(80, 131))
        A = rng.standard_normal((M, N))
        L = np.eye(N)

        G = gsvd(A, L, full_matrices=False)

        # Gamma_check corresponds to non-null part of A; compare to SVD singular values.
        _, svals, _ = np.linalg.svd(A, full_matrices=False)
        g_sorted = np.sort(G.gamma_check)[::-1]
        s_sorted = np.sort(svals)[::-1][: g_sorted.size]
        assert np.allclose(g_sorted, s_sorted, rtol=tol, atol=tol)
