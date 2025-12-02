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


def _project_out_colspace(B, z):
    if B.size == 0:
        return z
    P = B @ np.linalg.pinv(B)
    return z - P @ z


def test_oblique_pseudoinverses_against_explicit_and_properties():
    N = 100
    tol = 1e-10
    rng_global = np.random.default_rng(0)

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=False)

        # Explicit forms
        L_dag_A_explicit = (G.X2 @ (np.diag(1.0 / G.s_check) @ G.V2.T)) + (G.X3 @ G.V3.T)
        A_dag_L_explicit = (G.X2 @ (np.diag(1.0 / G.c_check) @ G.U2.T)) + (G.X1 @ G.U1.T)

        # API forms
        L_dag_A_api = G.get_L_oblique_pinv(matrix=True)
        A_dag_L_api = G.get_A_oblique_pinv(matrix=True)

        assert np.allclose(L_dag_A_explicit, L_dag_A_api, atol=tol)
        assert np.allclose(A_dag_L_explicit, A_dag_L_api, atol=tol)

        # Right-inverse on col(L)
        x0 = rng_global.standard_normal((A.shape[1],))
        z = L @ x0  # in col(L) subset of R^K
        z_recon = L @ (L_dag_A_api @ z)
        denom = max(1.0, np.linalg.norm(z))
        assert np.linalg.norm(z_recon - z) / denom <= tol

        # Annihilation on orthogonal complement of col(L)
        w = rng_global.standard_normal((L.shape[0],))
        z_orth = _project_out_colspace(G.Vhat, w)
        assert np.linalg.norm(L_dag_A_api @ z_orth) <= tol * max(1.0, np.linalg.norm(z_orth))

        # Relations AL_dag and LA_dag
        AL_dag = A @ L_dag_A_api
        LA_dag = L @ A_dag_L_api
        assert np.allclose(AL_dag, G.U2 @ (np.diag(G.gamma_check) @ G.V2.T), atol=tol)
        assert np.allclose(LA_dag, G.V2 @ (np.diag(1.0 / G.gamma_check) @ G.U2.T), atol=tol)
