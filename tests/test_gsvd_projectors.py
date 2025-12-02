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


def _span_project(B, v):
    return B @ np.linalg.pinv(B) @ v if B.size else np.zeros_like(v)


def _operator_to_dense(op, n):
    # Build a dense matrix from a LinearOperator by applying to basis vectors
    E = np.eye(n)
    return np.column_stack([op @ E[:, i] for i in range(n)])


def test_orthogonal_projectors_dense_and_operator():
    N = 100
    tol = 1e-10

    subspaces = [
        "col(A)",
        "col(A.T)",
        "ker(A)",
        "ker(A.T)",
        "col(L)",
        "col(L.T)",
        "ker(L)",
        "ker(L.T)",
    ]

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=False)

        # Bases for membership/orthogonality checks
        basis = {
            "col(A)": G.Uhat,
            "col(L)": G.Vhat,
            "ker(A)": G.X3,
            "ker(L)": G.X1,
            "col(A.T)": np.hstack([G.Y1, G.Y2]),
            "col(L.T)": np.hstack([G.Y2, G.Y3]),
            "ker(A.T)": G.Uhat,  # orthogonality checked via Uhat.T @ r ~ 0
            "ker(L.T)": G.Vhat,  # orthogonality checked via Vhat.T @ r ~ 0
        }

        for subspace in subspaces:
            P_mat = G.get_orthogonal_projector(subspace, matrix=True)

            # Idempotent and symmetric
            assert np.allclose(P_mat @ P_mat, P_mat, atol=tol)
            assert np.allclose(P_mat.T, P_mat, atol=tol)

            # Random vector checks
            rng = np.random.default_rng(seed + hash(subspace) % 1000)
            x_dim = P_mat.shape[1]
            x = rng.standard_normal((x_dim,))
            proj = P_mat @ x

            # Projection stability: projecting again does not change it
            assert np.allclose(P_mat @ proj, proj, atol=tol)

            B = basis[subspace]
            if subspace in {"col(A)", "col(L)", "col(A.T)", "col(L.T)"}:
                # proj should live in span(B)
                proj_B = _span_project(B, proj)
                denom = max(1.0, np.linalg.norm(proj))
                assert np.linalg.norm(proj - proj_B) / denom <= tol

                # residual should be orthogonal to B
                res = x - proj
                if B.size:
                    assert np.linalg.norm(B.T @ res) <= tol * max(1.0, np.linalg.norm(res))
                else:
                    # proj in kernel span
                    proj_B = _span_project(B, proj)
                    denom = max(1.0, np.linalg.norm(proj))
                    assert np.linalg.norm(proj - proj_B) / denom <= tol

                    # residual should be orthogonal to the kernel basis itself
                    res = x - proj
                    if B.size:
                        assert np.linalg.norm(B.T @ res) <= tol * max(1.0, np.linalg.norm(res))

            # LinearOperator vs dense
            P_op = G.get_orthogonal_projector(subspace, matrix=False)
            assert np.allclose(P_op @ x, proj, atol=tol)

            # Full dense equivalence check (costly but manageable for N=100)
            P_op_dense = _operator_to_dense(P_op, P_mat.shape[1])
            assert np.allclose(P_op_dense, P_mat, atol=1e-8)
