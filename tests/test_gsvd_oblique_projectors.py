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


def test_oblique_projectors_match_explicit_and_idempotent():
    N = 100
    tol = 1e-10

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=False)

        # Map which->(explicit_matrix, basis_keep, basis_kill)
        projector_defs = {
            1: (G.X1 @ G.Y1.T, G.X1, np.hstack([G.X2, G.X3])),
            2: (G.X2 @ G.Y2.T + G.X3 @ G.Y3.T, np.hstack([G.X2, G.X3]), G.X1),
            3: (G.X3 @ G.Y3.T, G.X3, np.hstack([G.X1, G.X2])),
            4: (G.X1 @ G.Y1.T + G.X2 @ G.Y2.T, np.hstack([G.X1, G.X2]), G.X3),
        }

        for which, (E_explicit, basis_keep, basis_kill) in projector_defs.items():
            E_api = G.get_oblique_projector(which=which, matrix=True)

            # Match explicit form
            assert np.allclose(E_api, E_explicit, atol=tol)

            # Idempotent
            assert np.allclose(E_api @ E_api, E_api, atol=tol)

            # Action on kept basis: E x = x
            if basis_keep.size:
                assert np.linalg.norm(E_api @ basis_keep - basis_keep, ord='fro') <= tol

            # Action on killed basis: E y = 0
            if basis_kill.size:
                assert np.linalg.norm(E_api @ basis_kill, ord='fro') <= tol

            # Operator form matches matrix form
            E_op = G.get_oblique_projector(which=which, matrix=False)
            rng = np.random.default_rng(seed + which)
            x = rng.standard_normal((N,))
            assert np.allclose(E_op @ x, E_api @ x, atol=tol)
