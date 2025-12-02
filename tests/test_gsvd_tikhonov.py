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


def test_tikhonov_via_gsvd_formulas():
    N = 100
    tol = 1e-8

    for seed in range(20):
        rng = np.random.default_rng(seed)
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        b = rng.standard_normal((A.shape[0],))
        lam = 10 ** rng.uniform(-6, 6)  # positive scalar across many decades

        G = gsvd(A, L, full_matrices=False)

        # Direct Tikhonov solution: (A^T A + λ L^T L) x = A^T b
        normal_mat = (A.T @ A) + lam * (L.T @ L)
        x_direct = np.linalg.solve(normal_mat, A.T @ b)

        # Formula (32): x* = L_A^† argmin_z ||A L_A^† z - b||^2 + λ||z||^2 + X1 U1^T b
        L_dag_A = G.get_L_oblique_pinv(matrix=True)  # N x K
        AL_dag = A @ L_dag_A                          # M x K
        # Solve z_star = argmin ||AL_dag z - b||^2 + λ||z||^2 -> (AL_dag^T AL_dag + λ I) z = AL_dag^T b
        z_normal = (AL_dag.T @ AL_dag) + lam * np.eye(AL_dag.shape[1])
        z_star = np.linalg.solve(z_normal, AL_dag.T @ b)
        x_formula_32 = (L_dag_A @ z_star) + (G.X1 @ (G.U1.T @ b))

        # Formula (33): x* = A_L^† argmin_z ||z - b||^2 + λ||L A_L^† z||^2
        A_dag_L = G.get_A_oblique_pinv(matrix=True)  # N x M
        LA_dag = L @ A_dag_L                          # K x M
        z2_normal = np.eye(A.shape[0]) + lam * (LA_dag.T @ LA_dag)
        z2_star = np.linalg.solve(z2_normal, b)
        x_formula_33 = A_dag_L @ z2_star

        assert np.allclose(x_direct, x_formula_32, atol=tol, rtol=tol)
        assert np.allclose(x_direct, x_formula_33, atol=tol, rtol=tol)
