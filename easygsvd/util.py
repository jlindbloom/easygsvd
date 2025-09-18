import numpy as np
import scipy.linalg as splinalg
from scipy.linalg import null_space



def complete_orthogonal(Qhat):
    """
    Given m x n Qhat with orthonormal columns, return
    (Qtilde, Qperp) where Qtilde = [Qhat, Qperp] is m x m orthogonal.
    """
    Qhat = np.asarray(Qhat)
    m, n = Qhat.shape
    Qperp = null_space(Qhat.T)
    Qtilde = np.hstack([Qhat, Qperp])  
    return Qtilde, Qperp





def random_rank_matrix(m, n, r, *, singular_values=None, seed=None, verify=False, return_factors=False):
    """
    Generate an m x n matrix with exact rank r.

    Parameters
    ----------
    m, n : int
        Desired matrix shape.
    r : int
        Desired rank. Must satisfy 0 <= r <= min(m, n).
    singular_values : array_like, optional
        Length-r positive values to use as singular values. If None, random
        values in (0.5, 2.0) are used.
    seed : int, optional
        Seed for reproducibility.
    verify : bool, default False
        If True, checks np.linalg.matrix_rank(A) == r and raises if not.
    return_factors : bool, default False
        If True, returns (A, U, s, V) where A = U @ diag(s) @ V.T,
        U is (m, r) with orthonormal columns, V is (n, r) with orthonormal
        columns, and s is length-r.

    Returns
    -------
    A : ndarray of shape (m, n)
        The constructed matrix.
    (optional) U, s, V : as described above.

    Raises
    ------
    ValueError
        If r is not feasible for the given shape or any singular value <= 0.
    """
    if not (isinstance(m, int) and isinstance(n, int) and isinstance(r, int)):
        raise ValueError("m, n, and r must be integers.")
    if m < 0 or n < 0:
        raise ValueError("m and n must be nonnegative.")
    if r < 0 or r > min(m, n):
        raise ValueError(f"Rank r={r} is impossible for shape ({m}, {n}). "
                         f"It must satisfy 0 <= r <= min(m, n) = {min(m, n)}.")

    rng = np.random.default_rng(seed)

    if r == 0:
        A = np.zeros((m, n), dtype=float)
        if return_factors:
            U = np.zeros((m, 0), dtype=float)
            s = np.zeros((0,), dtype=float)
            V = np.zeros((n, 0), dtype=float)
            return A, U, s, V
        return A

    # Orthonormal columns via QR
    QU, _ = np.linalg.qr(rng.standard_normal((m, r)), mode="reduced")  # (m, r)
    QV, _ = np.linalg.qr(rng.standard_normal((n, r)), mode="reduced")  # (n, r)

    if singular_values is None:
        s = rng.uniform(0.5, 2.0, size=r)
    else:
        s = np.asarray(singular_values, dtype=float)
        if s.shape != (r,):
            raise ValueError(f"singular_values must have shape ({r},), got {s.shape}.")
        if np.any(s <= 0):
            raise ValueError("All singular values must be strictly positive.")

    # Construct A = U diag(s) V^T
    A = QU @ (s[:, None] * QV.T)  # Equivalent to QU @ np.diag(s) @ QV.T but faster

    if verify:
        rank = np.linalg.matrix_rank(A)
        if rank != r:
            raise RuntimeError(f"Constructed matrix has rank {rank}, expected {r}.")

    if return_factors:
        return A, QU, s, QV
    return A




def _orthonormal_basis(A, svd_tol=None):
    """
    Return Q with orthonormal columns spanning col(A),
    using an SVD-based rank decision.
    """
    if A.size == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if svd_tol is None:
        # Standard numerical rank tolerance
        svd_tol = np.finfo(s.dtype).eps * max(A.shape) * (s[0] if s.size else 0.0)
    r = int(np.sum(s > svd_tol))
    return U[:, :r] if r > 0 else np.zeros((A.shape[0], 0), dtype=A.dtype)

def colspaces_equal(A, B, tol=1e-10, svd_tol=None, return_diagnostics=False):
    """
    Determine whether col(A) == col(B) by testing if the spectral norm
    ||Q_A Q_A^T - Q_B Q_B^T||_2 <= tol, where Q_* have orthonormal columns
    spanning the respective column spaces.

    Parameters
    ----------
    A, B : array_like (m x n_a), (m x n_b)
        Input matrices with the SAME number of rows m.
    tol : float
        Tolerance for deciding equality of subspaces via the projector difference.
    svd_tol : float or None
        Numerical rank threshold used to determine the basis size from SVD
        (defaults to eps * max(m, n) * largest_singular_value).
    return_diagnostics : bool
        If True, also return a dict with details (norm value, ranks, etc.).

    Returns
    -------
    equal : bool
        True iff the projector-difference norm <= tol.
    info : dict (optional)
        Contains:
          - 'norm': spectral norm of P_A - P_B
          - 'rank_A', 'rank_B'
          - 'm': number of rows
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"A and B must have the same number of rows; got {A.shape[0]} and {B.shape[0]}.")

    QA = _orthonormal_basis(A, svd_tol=svd_tol)
    QB = _orthonormal_basis(B, svd_tol=svd_tol)

    # Orthogonal projectors onto col(A) and col(B)
    PA = QA @ QA.T if QA.shape[1] > 0 else np.zeros((A.shape[0], A.shape[0]), dtype=A.dtype)
    PB = QB @ QB.T if QB.shape[1] > 0 else np.zeros((B.shape[0], B.shape[0]), dtype=B.dtype)

    # Spectral norm ||PA - PB||_2 = largest singular value
    # np.linalg.norm(M, 2) returns the spectral norm for a matrix
    norm_diff = np.linalg.norm(PA - PB, 2)

    equal = norm_diff <= tol
    if return_diagnostics:
        return equal, {
            "norm": float(norm_diff),
            "rank_A": int(QA.shape[1]),
            "rank_B": int(QB.shape[1]),
            "m": int(A.shape[0]),
            "tol": float(tol),
        }
    return equal



def colspace_intersection_AT_LT(A, L, tol_rank=1e-12, tol_sv=1-1e-12):
    """
    Return Z whose columns form an orthonormal basis for col(A^T) ∩ col(L^T).

    Parameters
    ----------
    A, L : np.ndarray
        Shapes (m, n) and (p, n). The intersection lives in R^n.
    tol_rank : float
        Rank tolerance for SVD truncation of A^T and L^T.
    tol_sv : float
        Threshold for treating a singular value as '1'. Use something like 1-1e-12.

    Returns
    -------
    Z : np.ndarray of shape (n, k)
        Orthonormal columns spanning the intersection (k can be 0).
    singvals : np.ndarray
        Singular values of Q_A^T Q_L (principal cosines), useful for diagnostics.
    """
    # Orthonormal bases for col(A^T) and col(L^T)
    UA, sA, _ = np.linalg.svd(A.T, full_matrices=False)
    rA = np.sum(sA > tol_rank * (sA[0] if sA.size else 1.0))
    QA = UA[:, :rA]  # (n, rA)

    UL, sL, _ = np.linalg.svd(L.T, full_matrices=False)
    rL = np.sum(sL > tol_rank * (sL[0] if sL.size else 1.0))
    QL = UL[:, :rL]  # (n, rL)

    if QA.size == 0 or QL.size == 0:
        return np.empty((A.shape[1], 0)), np.array([])

    # Overlap SVD
    M = QA.T @ QL                       # (rA, rL)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Indices where singular values ≈ 1
    idx = np.where(s >= tol_sv)[0]

    if idx.size == 0:
        return np.empty((A.shape[1], 0)), s

    # Basis for the intersection (either expression is fine)
    Z = QA @ U[:, idx]                  # (n, k)
    # Optionally: Z = QL @ Vt.T[:, idx]

    return Z, s



def complete_orthogonal(Qhat, tol=1e-12):
    """
    Given m x n Qhat with orthonormal columns, return
    (Qtilde, Qperp) where Qtilde = [Qhat, Qperp] is m x m orthogonal.
    """
    Qhat = np.asarray(Qhat)
    m, n = Qhat.shape
    # Optional sanity check
    if not np.allclose(Qhat.T @ Qhat, np.eye(n), atol=10*tol):
        raise ValueError("Qhat columns are not orthonormal within tolerance.")
    # Orthonormal basis of null( Qhat^T ) is the orthogonal complement
    Qperp = null_space(Qhat.T, rcond=tol)          # shape (m, m-n), orthonormal
    Qtilde = np.hstack([Qhat, Qperp])              # shape (m, m)
    return Qtilde, Qperp



def colspaces_direct_sum_RN(X1, X2, X3, tol=1e-10, svd_tol=None, return_diagnostics=False):
    """
    Check whether R^N = col(X1) ⊕ col(X2) ⊕ col(X3).

    Logic:
      1) Let Q_i be orthonormal bases for col(X_i), with ranks r_i.
      2) Necessary: r1 + r2 + r3 == N (dimensions add up to N).
      3) Directness + spanning: columns of [Q1 Q2 Q3] are linearly independent,
         i.e., rank([Q1 Q2 Q3]) = N (within tolerance).

    Parameters
    ----------
    X1, X2, X3 : array_like, each shape (N, k_i)
        Input matrices (same row count N), columns span the subspaces.
    tol : float
        Tolerance for the full-rank test on the concatenated basis.
    svd_tol : float or None
        Tolerance for numerical rank when forming Q_i via SVD.
    return_diagnostics : bool
        If True, also return a dict with details.

    Returns
    -------
    ok : bool
        True iff R^N is the direct sum of the three column spaces.
    info : dict (optional)
        'ranks' : (r1, r2, r3)
        'N' : N
        'dim_sum_equals_N' : bool
        'min_singular_concat' : smallest singular value of [Q1 Q2 Q3]
        'rank_concat' : numerical rank of [Q1 Q2 Q3]
        'tol_used' : tolerance used for the full-rank test
    """
    X1 = np.asarray(X1); X2 = np.asarray(X2); X3 = np.asarray(X3)
    N = X1.shape[0]
    if X2.shape[0] != N or X3.shape[0] != N:
        raise ValueError("All inputs must have the same number of rows N.")

    # Orthonormal bases for each column space
    Q1 = _orthonormal_basis(X1, svd_tol=svd_tol)
    Q2 = _orthonormal_basis(X2, svd_tol=svd_tol)
    Q3 = _orthonormal_basis(X3, svd_tol=svd_tol)
    r1, r2, r3 = Q1.shape[1], Q2.shape[1], Q3.shape[1]

    # Quick necessary check: dimensions must add to N
    dim_sum_equals_N = (r1 + r2 + r3 == N)
    if not dim_sum_equals_N:
        if return_diagnostics:
            return False, {
                "ranks": (r1, r2, r3),
                "N": N,
                "dim_sum_equals_N": False,
                "min_singular_concat": 0.0,
                "rank_concat": r1 + r2 + r3,   # rank cannot exceed this
                "tol_used": float(tol),
            }
        return False

    # Concatenate the orthonormal bases
    Q = np.hstack([Q1, Q2, Q3])  # shape (N, N) if dim_sum_equals_N

    # Full-rank test via SVD of the concatenated basis
    s = np.linalg.svd(Q, compute_uv=False)
    # Use a standard relative threshold for rank
    tol_used = tol if tol is not None else (np.finfo(s.dtype).eps * N * (s[0] if s.size else 0.0))
    rankQ = int(np.sum(s > tol_used))
    ok = (rankQ == N)

    if return_diagnostics:
        return ok, {
            "ranks": (r1, r2, r3),
            "N": N,
            "dim_sum_equals_N": True,
            "min_singular_concat": float(s[-1]) if s.size else 0.0,
            "rank_concat": rankQ,
            "tol_used": float(tol_used),
        }
    return ok