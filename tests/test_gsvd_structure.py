import numpy as np

from easygsvd.gsvd import gsvd


def _random_full_column_rank_pair(N, seed, max_tries=10):
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        M = int(rng.integers(80, 131))  # around 100, may be below N
        K = int(rng.integers(80, 131))
        A = rng.standard_normal((M, N))
        L = rng.standard_normal((K, N))
        if np.linalg.matrix_rank(np.vstack([A, L])) == N:
            return A, L
    raise RuntimeError("Failed to generate full-column-rank stacked matrix within max_tries.")


def test_gsvd_shapes_orthogonality_and_ordering():
    tol = 1e-12
    ortho_tol = 1e-10
    N = 100

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=True)

        M, N_actual = A.shape
        K = L.shape[0]
        assert N_actual == N
        r_intersect = G.r_int

        # Shape checks
        assert G.U1.shape == (M, G.n_L)
        assert G.U2.shape == (M, r_intersect)
        assert G.V2.shape == (K, r_intersect)
        assert G.V3.shape == (K, G.n_A)
        assert G.X1.shape == (N, G.n_L)
        assert G.X2.shape == (N, r_intersect)
        assert G.X3.shape == (N, G.n_A)

        # Orthogonality of hatted bases
        assert np.allclose(G.Uhat.T @ G.Uhat, np.eye(G.Uhat.shape[1]), atol=ortho_tol)
        assert np.allclose(G.Vhat.T @ G.Vhat, np.eye(G.Vhat.shape[1]), atol=ortho_tol)

        # Full orthogonality when full_matrices=True
        assert np.allclose(G.U.T @ G.U, np.eye(M), atol=ortho_tol)
        assert np.allclose(G.V.T @ G.V, np.eye(K), atol=ortho_tol)

        # Biorthogonality of X and Y
        assert np.allclose(G.X @ G.Y.T, np.eye(N), atol=ortho_tol)
        assert np.allclose(G.Y @ G.X.T, np.eye(N), atol=ortho_tol)

        # Generalized singular values consistency
        assert np.allclose(G.c**2 + G.s**2, np.ones_like(G.c), atol=1e-10)

        # Ordering
        assert np.all(np.diff(G.c) <= 1e-12)
        assert np.all(np.diff(G.s) >= -1e-12)

        # Intersection gamma_check relation
        assert np.allclose(G.gamma_check, G.c_check / G.s_check, atol=1e-12, rtol=1e-12)
