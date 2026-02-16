import numpy as np

from easygsvd.gsvd import gsvd


def _random_full_column_rank_pair_complex(N, seed, max_tries=20):
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        M = int(rng.integers(N + 5, N + 15))
        K = int(rng.integers(N + 5, N + 15))
        A = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
        L = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
        if np.linalg.matrix_rank(np.vstack([A, L])) == N:
            return A, L
    raise RuntimeError("Failed to generate full-column-rank stacked matrix within max_tries.")


def _operator_to_dense(op, n, dtype):
    E = np.eye(n, dtype=dtype)
    return np.column_stack([op @ E[:, i] for i in range(n)])


def test_gsvd_reconstructs_complex_A_and_L():
    N = 20
    tol = 1e-9

    for seed in range(6):
        A, L = _random_full_column_rank_pair_complex(N=N, seed=seed)

        # Full decomposition
        G_full = gsvd(A, L, full_matrices=True)
        Y12 = np.hstack([G_full.Y1, G_full.Y2])  # shape N x r_A
        Y23 = np.hstack([G_full.Y2, G_full.Y3])  # shape N x r_L

        A_recon_full = G_full.Uhat @ (np.diag(G_full.c_hat) @ Y12.conj().T)
        L_recon_full = G_full.Vhat @ (np.diag(G_full.s_hat) @ Y23.conj().T)

        assert np.allclose(A_recon_full, A, atol=tol)
        assert np.allclose(L_recon_full, L, atol=tol)

        # Economic form reconstruction
        G = gsvd(A, L, full_matrices=False)
        A_recon = (G.U1 @ G.Y1.conj().T) + (G.U2 @ (np.diag(G.c_check) @ G.Y2.conj().T))
        L_recon = (G.V2 @ (np.diag(G.s_check) @ G.Y2.conj().T)) + (G.V3 @ G.Y3.conj().T)

        assert np.allclose(A_recon, A, atol=tol)
        assert np.allclose(L_recon, L, atol=tol)


def test_complex_orthogonal_projectors_are_hermitian():
    N = 18
    tol = 1e-10

    for seed in range(4):
        A, L = _random_full_column_rank_pair_complex(N=N, seed=100 + seed)
        G = gsvd(A, L, full_matrices=False)

        for subspace in ["col(A)", "ker(L.T)"]:
            P_mat = G.get_orthogonal_projector(subspace, matrix=True)

            # Idempotent and Hermitian
            assert np.allclose(P_mat @ P_mat, P_mat, atol=tol)
            assert np.allclose(P_mat.conj().T, P_mat, atol=tol)

            rng = np.random.default_rng(200 + seed)
            x = rng.standard_normal((P_mat.shape[1],)) + 1j * rng.standard_normal((P_mat.shape[1],))
            proj = P_mat @ x
            assert np.allclose(P_mat @ proj, proj, atol=tol)

            # LinearOperator vs dense
            P_op = G.get_orthogonal_projector(subspace, matrix=False)
            assert np.allclose(P_op @ x, proj, atol=tol)

            P_op_dense = _operator_to_dense(P_op, P_mat.shape[1], P_mat.dtype)
            assert np.allclose(P_op_dense, P_mat, atol=1e-8)
