import numpy as np
from scipy.sparse.linalg import aslinearoperator
import scipy.sparse as sps
from scipy.linalg import null_space


class GSVDResult:
    """Provides interface to the GSVD decomposition.
    """

    def __init__(self, A, L, Uhat, Vhat, X, Y, c, s,
                 r_A, r_L, n_A, n_L, Uperp=None, Vperp=None):
        
        # Bind GSVD quantities
        self.A = A
        self.L = L
        self.Uhat = Uhat
        self.Vhat = Vhat
        self.Y = Y
        self.X = X
        self.c = c
        self.s = s
        self.n_A = n_A
        self.n_L = n_L
        self.r_A = r_A
        self.r_L = r_L
        self.r_int = self.A.shape[1] - self.n_A - self.n_L

        # Partition some quantities 
        self.U1 = self.Uhat[:, :n_L]          # cols for ker(L)
        self.U2 = self.Uhat[:, n_L:]          # remaining cols
        self.V2 = self.Vhat[:, :r_A - n_L]    # cols for ker(A)^⊥_L
        self.V3 = self.Vhat[:, r_A - n_L:]    # cols for ker(A)
        self.X1 = self.X[:, :n_L]
        self.X2 = self.X[:, n_L:r_A]
        self.X3 = self.X[:, r_A:]
        self.Y1 = self.Y[:, :n_L]
        self.Y2 = self.Y[:, n_L:r_A]
        self.Y3 = self.Y[:, r_A:]
        self.c_hat = self.c[:r_A]
        self.s_hat = self.s[n_L:]
        self.c_check = self.c[n_L:r_A]
        self.s_check = self.s[n_L:r_A]

        # Handle full decomposition
        self._Uperp = Uperp
        self._Vperp = Vperp

        # Define full U and V if Uperp and Vperp provided
        self.complete = True
        if self._Uperp is not None:
            self._U = np.hstack([self.Uhat, self._Uperp])
        else:
            self.complete = False
        if self._Vperp is not None:
            self._V = np.hstack([self.Vhat, self._Vperp])
        else:
            self.complete = False

        # Generalized singular values
        self.gamma_check = self.c_check / self.s_check
        self.gamma = np.hstack([
            np.inf * np.ones(self.n_L),
            self.gamma_check,
            np.zeros(self.r_A)
        ])

        # Internal: workspace / tuning parameters (set by gsvd / _GSVDWorkspace)
        self._workspace = None
        self._tol = None
        self._full_matrices = None

    # -------- Properties for full U, V, Uperp, Vperp --------

    @property
    def Uperp(self):
        if self._Uperp is None:
            raise AttributeError("Must perform GSVD with full_matrices=True")
        return self._Uperp
    
    @property
    def Vperp(self):
        if self._Vperp is None:
            raise AttributeError("Must perform GSVD with full_matrices=True")
        return self._Vperp
    
    @property
    def U(self):
        if not self.complete:
            raise AttributeError("Must perform GSVD with full_matrices=True")
        return self._U

    @property
    def V(self):
        if not self.complete:
            raise AttributeError("Must perform GSVD with full_matrices=True")
        return self._V

    
    
    # -------- Projectors and pseudoinverses --------

    def get_orthogonal_projector(self, subspace, matrix=True):
        """Returns the orthogonal projector onto the specified subspace.

        subspace: which subspace to consider.
        matrix: if True, returns the results as dense matrices.
                If False, uses implicit LinearOperators. 
        """
        valid_subspaces = [
            "col(A)", "col(A.T)", "ker(A)", "ker(A.T)", 
            "col(L)", "col(L.T)", "ker(L)", "ker(L.T)"
        ]
        assert subspace in valid_subspaces, \
            f"Invalid subspace, must be one of: {valid_subspaces}"

        if subspace == "col(A)":
            if matrix:
                return self.Uhat @ self.Uhat.T
            else:
                return aslinearoperator(self.Uhat) @ aslinearoperator(self.Uhat.T)

        elif subspace == "col(A.T)":
            Z = np.hstack([self.Y1, self.Y2])
            Q, R = np.linalg.qr(Z, mode="reduced")
            if matrix:
                return Q @ Q.T
            else:
                return aslinearoperator(Q) @ aslinearoperator(Q.T)

        elif subspace == "ker(A)":
            Q, R = np.linalg.qr(self.X3, mode="reduced")
            if matrix:
                return Q @ Q.T
            else:
                return aslinearoperator(Q) @ aslinearoperator(Q.T)

        elif subspace == "ker(A.T)":
            if matrix:
                return np.eye(self.Uhat.shape[0]) - (self.Uhat @ self.Uhat.T)
            else:
                eye_op = aslinearoperator(sps.diags(np.ones(self.Uhat.shape[0])))
                return eye_op - (aslinearoperator(self.Uhat) @ aslinearoperator(self.Uhat.T))

        elif subspace == "col(L)":
            if matrix:
                return self.Vhat @ self.Vhat.T
            else:
                return aslinearoperator(self.Vhat) @ aslinearoperator(self.Vhat.T)

        elif subspace == "col(L.T)":
            Z = np.hstack([self.Y2, self.Y3])
            Q, R = np.linalg.qr(Z, mode="reduced")
            if matrix:
                return Q @ Q.T
            else:
                return aslinearoperator(Q) @ aslinearoperator(Q.T)

        elif subspace == "ker(L)":
            Q, R = np.linalg.qr(self.X1, mode="reduced")
            if matrix:
                return Q @ Q.T
            else:
                return aslinearoperator(Q) @ aslinearoperator(Q.T)

        elif subspace == "ker(L.T)":
            if matrix:
                return np.eye(self.Vhat.shape[0]) - (self.Vhat @ self.Vhat.T)
            else:
                eye_op = aslinearoperator(sps.diags(np.ones(self.Vhat.shape[0])))
                return eye_op - (aslinearoperator(self.Vhat) @ aslinearoperator(self.Vhat.T))

        else:
            raise NotImplementedError
        

    def get_oblique_projector(self, which=1, matrix=True):
        """Returns the oblique projector related to the two subspaces.

        which: 1-4, determines which oblique projector to return.
        matrix: if True, returns the results as dense matrices.
                If False, uses implicit LinearOperators. 
        """
        valid_options = [1, 2, 3, 4]
        assert which in valid_options, \
            f"Invalid option, must be one of {valid_options}"

        if which == 1:
            # projection onto ker(L) along ker(L)^{perp_A}
            if matrix:
                return self.X1 @ self.Y1.T
            else:
                return aslinearoperator(self.X1) @ aslinearoperator(self.Y1.T)

        elif which == 2:
            # projection onto ker(L)^{perp_A} along ker(L)
            if matrix:
                return (self.X2 @ self.Y2.T) + (self.X3 @ self.Y3.T)
            else:
                return (
                    aslinearoperator(self.X2) @ aslinearoperator(self.Y2.T)
                    + aslinearoperator(self.X3) @ aslinearoperator(self.Y3.T)
                )

        elif which == 3:
            # projection onto ker(A) along ker(A)^{perp_L}
            if matrix:
                return self.X3 @ self.Y3.T
            else:
                return aslinearoperator(self.X3) @ aslinearoperator(self.Y3.T)

        elif which == 4:
            # projection onto ker(A)^{perp_L} along ker(A)
            if matrix:
                return (self.X1 @ self.Y1.T) + (self.X2 @ self.Y2.T)
            else:
                return (
                    aslinearoperator(self.X1) @ aslinearoperator(self.Y1.T)
                    + aslinearoperator(self.X2) @ aslinearoperator(self.Y2.T)
                )

        else:
            raise NotImplementedError


    def get_L_oblique_pinv(self, matrix=True):
        r"""Returns the oblique (A-weighted) pseudoinverse \(L_A^\dagger\)."""
        if matrix:
            Lopinv = (self.X2 @ (np.diag(1.0 / self.s_check) @ self.V2.T)) \
                     + (self.X3 @ self.V3.T)
        else:
            Lopinv = (
                aslinearoperator(self.X2)
                @ aslinearoperator(sps.diags(1.0 / self.s_check))
                @ aslinearoperator(self.V2.T)
            ) + (aslinearoperator(self.X3) @ aslinearoperator(self.V3.T))

        return Lopinv
    

    def get_A_oblique_pinv(self, matrix=True):
        r"""Returns the oblique pseudoinverse \(A_L^\dagger\)."""
        if matrix:
            Lopinv = (self.X2 @ (np.diag(1.0 / self.c_check) @ self.U2.T)) \
                     + (self.X1 @ self.U1.T)
        else:
            Lopinv = (
                aslinearoperator(self.X2)
                @ aslinearoperator(sps.diags(1.0 / self.c_check))
                @ aslinearoperator(self.U2.T)
            ) + (aslinearoperator(self.X1) @ aslinearoperator(self.U1.T))

        return Lopinv
    

    def get_L_standard_form_data(self, matrix=True):
        r"""Compute and return some quantities related to the oblique pseudoinverse.

        Returns
        -------
        E      : oblique projection matrix s.t. L_A^\dagger = E L^\dagger
        Lopinv : oblique (A-weighted) pseudoinverse L_A^\dagger
        ALopinv: A @ L_A^\dagger
        kermat : matrix whose columns span ker(L)
        """
        if matrix:
            E = np.eye(self.X1.shape[0]) - self.X1 @ (self.U1.T @ self.A)
            Lopinv = self.get_L_oblique_pinv(matrix=True)
            ALopinv = self.U2 @ (np.diag(self.gamma_check) @ self.V2.T)
            kermat = self.X1
        else:
            E = aslinearoperator(sps.diags(np.ones(self.X1.shape[0]))) - (
                aslinearoperator(self.X1)
                @ aslinearoperator((self.A.T @ self.U1).T)
            )
            Lopinv = self.get_L_oblique_pinv(matrix=False)
            ALopinv = (
                aslinearoperator(self.U2)
                @ aslinearoperator(sps.diags(self.gamma_check))
                @ aslinearoperator(self.V2.T)
            )
            kermat = aslinearoperator(self.X1)

        return E, Lopinv, ALopinv, kermat

    # ---------- Internal clone helper ----------

    def _clone(self):
        """Return a NEW GSVDResult object with identical factors."""
        new = GSVDResult(
            self.A, self.L,
            self.Uhat, self.Vhat,
            self.X, self.Y,
            self.c, self.s,
            self.r_A, self.r_L, self.n_A, self.n_L,
            Uperp=self._Uperp, Vperp=self._Vperp
        )
        new._workspace = self._workspace
        new._tol = self._tol
        new._full_matrices = self._full_matrices
        return new

    # ---------- Incremental API helpers ----------

    def _normalize_new_columns(self, base_mat, new_cols):
        """Normalize a proposed block of new columns.

        base_mat : existing A or L, shape (m, n)
        new_cols : None, 1D array (m,), or 2D (m, p)

        Returns
        -------
        norm_cols : None or ndarray, shape (m, p_new)
                    (p_new >= 1 if not None)
        p_new     : int, number of new columns (0 if None)
        """
        if new_cols is None:
            return None, 0

        arr = np.asarray(new_cols)
        m = base_mat.shape[0]

        # Handle 1D vector
        if arr.ndim == 1:
            if arr.size == 0:
                return None, 0
            if arr.shape[0] != m:
                raise ValueError(
                    f"New column length {arr.shape[0]} does not match "
                    f"existing matrix row dimension {m}."
                )
            arr = arr.reshape(m, 1)
            return arr, 1

        # Handle 2D matrix
        if arr.ndim == 2:
            if arr.shape[0] != m:
                raise ValueError(
                    f"New columns have {arr.shape[0]} rows but base matrix has {m}."
                )
            p = arr.shape[1]
            if p == 0:
                return None, 0
            return arr, p

        raise ValueError("new_cols must be None, 1D, or 2D array-like.")

    # ---------- Incremental API ----------

    def append_column(self, a, ell):
        """Return a NEW GSVDResult for ([A A_new], [L L_new]) via incremental update.

        Parameters
        ----------
        a   : array_like, None, or 2D array
            New column(s) for A, shape (M,) or (M, P_A),
            or None / shape (M, 0) meaning *no update to A*.
        ell : array_like, None, or 2D array
            New column(s) for L, shape (K,) or (K, P_L),
            or None / shape (K, 0) meaning *no update to L*.

        Rules
        -----
        - If A and L both get zero new columns (None or shape (., 0)):
            -> no-op; returns a NEW GSVDResult that is a clone of self.
        - If exactly one of them gets a positive number of new columns, or
          if the positive counts differ:
            -> raises ValueError (cannot change #cols of A and L differently).
        - If both get P > 0 new columns (same P):
            -> append all P columns, using incremental GSVD update.
        """
        # Normalize new columns for A and L
        A_new, pA = self._normalize_new_columns(self.A, a)
        L_new, pL = self._normalize_new_columns(self.L, ell)

        # Case 1: both effectively "no new columns"
        if pA == 0 and pL == 0:
            return self._clone()

        # Case 2: only one side gets new columns or the counts differ
        if pA != pL:
            raise ValueError(
                "A and L must receive the same positive number of new columns. "
                f"Got pA={pA}, pL={pL}."
            )

        P = pA  # == pL > 0

        # Case 3: append P columns to both A and L
        if (self._workspace is None) or (not isinstance(self._workspace, _GSVDWorkspace)):
            # Rebuild workspace from current A, L if needed
            tol = self._tol if self._tol is not None else 1e-12
            full_matrices = self._full_matrices if self._full_matrices is not None else False
            ws_base = _GSVDWorkspace(self.A, self.L, full_matrices=full_matrices, tol=tol)
        else:
            ws_base = self._workspace

        # Work on a copy so the original factorization is not mutated
        ws_new = ws_base.copy()

        # Append each new column sequentially
        for j in range(P):
            a_col = A_new[:, j]
            ell_col = L_new[:, j]
            ws_new.append_column(a_col, ell_col)

        res_new = ws_new.to_gsvd_result()
        return res_new


class _GSVDWorkspace:
    """Internal workspace for GSVD with support for incremental column updates.

    Not part of the public API. Users interact via GSVDResult and gsvd().
    """

    def __init__(self, A, L, full_matrices=False, tol=1e-12):
        A = np.asarray(A)
        L = np.asarray(L)
        M, N = A.shape
        K, N2 = L.shape
        assert N == N2, "A and L must have the same number of columns!"
        self.A = A
        self.L = L
        self.M = M
        self.K = K
        self.N = N
        self.full_matrices = full_matrices
        self.tol = tol

        # Cannot have full column rank if N > M+K
        if N > M + K:
            raise np.linalg.LinAlgError(
                "Stacked matrix [A; L] has fewer rows (M+K) than columns (N); "
                "cannot have full column rank."
            )

        # QR of stacked matrix
        stack_matrix = np.vstack([A, L])
        Q, R = np.linalg.qr(stack_matrix, mode="reduced")
        self.Q = Q
        self.R = R
        self.Q_A = Q[:M, :]
        self.Q_L = Q[M:, :]

        # Enforce full column rank of stacked [A; L] via R's diagonal
        self._check_full_column_rank_from_R(R)

        # Gram matrix GA = Q_A^T Q_A
        GA = self.Q_A.T @ self.Q_A
        self.GA = GA

        # Eigendecomposition of GA
        c_sq, W = np.linalg.eigh(GA)
        idx = np.argsort(c_sq)[::-1]
        c_sq = c_sq[idx]
        W = W[:, idx]

        # numerical safety: eigenvalues should be in [0, 1]
        c_sq = np.clip(c_sq.real, 0.0, 1.0)
        s_sq = 1.0 - c_sq
        s_sq = np.clip(s_sq, 0.0, 1.0)

        self.c_sq = c_sq
        self.s_sq = s_sq
        self.c = np.sqrt(c_sq)
        self.s = np.sqrt(s_sq)
        self.W = W

    def _check_full_column_rank_from_R(self, R):
        """Check that the stacked matrix [A; L] has full column rank N.

        Uses only the diagonal of the R factor from QR.
        """
        diagR = np.abs(np.diag(R))
        if diagR.size == 0:
            raise np.linalg.LinAlgError(
                "Stacked matrix [A; L] has no columns; cannot form GSVD."
            )
        max_diag = diagR.max()
        if max_diag == 0.0:
            raise np.linalg.LinAlgError(
                "Stacked matrix [A; L] is rank deficient (all diagonal entries of R are zero)."
            )
        # Require every diagonal to be "large enough" relative to the largest
        if np.min(diagR) < self.tol * max_diag:
            raise np.linalg.LinAlgError(
                "Stacked matrix [A; L] is numerically rank-deficient; "
                "this GSVD implementation requires full column rank of [A; L]."
            )

    def copy(self):
        """Deep copy the workspace so we can update without mutating the original."""
        new = _GSVDWorkspace.__new__(_GSVDWorkspace)

        # Copy scalars
        new.M = self.M
        new.K = self.K
        new.N = self.N
        new.full_matrices = self.full_matrices
        new.tol = self.tol

        # Copy arrays
        new.A = self.A.copy()
        new.L = self.L.copy()
        new.Q = self.Q.copy()
        new.R = self.R.copy()
        new.Q_A = self.Q_A.copy()
        new.Q_L = self.Q_L.copy()
        new.GA = self.GA.copy()
        new.c_sq = self.c_sq.copy()
        new.s_sq = self.s_sq.copy()
        new.c = self.c.copy()
        new.s = self.s.copy()
        new.W = self.W.copy()

        return new

    def append_column(self, a, ell, rank_tol=1e-14):
        """Append a new column (a, ell) to (A, L) and update QR and Q_A^T Q_A.

        a   : array_like or None (1D, length M)
        ell : array_like or None (1D, length K)

        - If both None: no-op.
        - If exactly one is None: raise ValueError.
        - Else: append column to both A and L, enforcing that the new
          stacked matrix [A; L] has full column rank.
        """
        # No-op
        if a is None and ell is None:
            return

        # Exactly one None -> invalid, don't silently pad
        if (a is None) != (ell is None):
            raise ValueError(
                "Cannot append a column to only A or only L. "
                "Either provide both a and ell, or both None."
            )

        # Now both a and ell are provided
        a = np.asarray(a, dtype=self.A.dtype).ravel()
        ell = np.asarray(ell, dtype=self.L.dtype).ravel()
        assert a.shape[0] == self.M, "a must have length M"
        assert ell.shape[0] == self.K, "ell must have length K"

        # Update A, L (true column append)
        self.A = np.column_stack([self.A, a])
        self.L = np.column_stack([self.L, ell])

        # New stacked column
        c_stack = np.concatenate([a, ell])
        Q = self.Q
        R = self.R
        M_plus_K, N = Q.shape
        assert M_plus_K == self.M + self.K

        # Project onto current Q
        y = Q.T @ c_stack
        r = c_stack - Q @ y
        rho = np.linalg.norm(r)
        c_norm = np.linalg.norm(c_stack) + 1e-32

        # Enforce that the new column is not in the span of existing columns
        if rho < rank_tol * c_norm:
            raise np.linalg.LinAlgError(
                "Appending column would make the stacked matrix [A; L] "
                "numerically rank-deficient."
            )

        q_new = r / rho

        # Update Q and R
        Q_new = np.column_stack([Q, q_new])
        R_new = np.zeros((N + 1, N + 1), dtype=R.dtype)
        R_new[:N, :N] = R
        R_new[:N, N] = y
        R_new[N, N] = rho

        self.Q = Q_new
        self.R = R_new

        # Update Q_A and Q_L
        q_A_new = q_new[:self.M]
        q_L_new = q_new[self.M:]

        Q_A_old = self.Q_A
        self.Q_A = np.column_stack([self.Q_A, q_A_new])
        self.Q_L = np.column_stack([self.Q_L, q_L_new])
        self.N = N + 1

        # Update GA with a bordered matrix
        GA_old = self.GA
        g = Q_A_old.T @ q_A_new  # cross terms with old columns
        gamma = float(q_A_new @ q_A_new)
        GA_new = np.zeros((N + 1, N + 1), dtype=GA_old.dtype)
        GA_new[:N, :N] = GA_old
        GA_new[:N, N] = g
        GA_new[N, :N] = g
        GA_new[N, N] = gamma
        self.GA = GA_new

        # Eigendecomposition of GA_new
        c_sq, W = np.linalg.eigh(GA_new)
        idx = np.argsort(c_sq)[::-1]
        c_sq = c_sq[idx]
        W = W[:, idx]

        c_sq = np.clip(c_sq.real, 0.0, 1.0)
        s_sq = 1.0 - c_sq
        s_sq = np.clip(s_sq, 0.0, 1.0)

        self.c_sq = c_sq
        self.s_sq = s_sq
        self.c = np.sqrt(c_sq)
        self.s = np.sqrt(s_sq)
        self.W = W

    def to_gsvd_result(self):
        """Build a GSVDResult object from the current workspace state."""
        A, L = self.A, self.L
        M, K, N = self.M, self.K, self.N
        c, s = self.c, self.s
        W = self.W
        tol = self.tol
        full_matrices = self.full_matrices
        Q_A, Q_L, R = self.Q_A, self.Q_L, self.R

        c_sq = c**2
        s_sq = s**2

        n_A = int(np.sum(c_sq < tol))
        r_A = N - n_A
        n_L = int(np.sum(s_sq < tol))
        r_L = N - n_L

        if r_A == 0:
            raise np.linalg.LinAlgError("Rank of A is zero in GSVD.")
        if N - n_L == 0:
            raise np.linalg.LinAlgError("No nonzero generalized singular values for L.")

        c_hat = c[:r_A]
        s_hat = s[n_L:]

        # X = R^{-1} W  (R should be nonsingular by our rank checks)
        X = np.linalg.solve(R, W)

        W_A_1 = W[:, :r_A]
        W_L_2 = W[:, n_L:]

        # Uhat, Vhat
        Uhat = Q_A @ (W_A_1 @ np.diag(1.0 / c_hat))
        Vhat = Q_L @ (W_L_2 @ np.diag(1.0 / s_hat))

        # Y = X^{-T}
        Y = np.linalg.solve(X.T, np.eye(X.shape[0], dtype=X.dtype))

        if full_matrices:
            Uperp = null_space(Uhat.T)
            Vperp = null_space(Vhat.T)
        else:
            Uperp = None
            Vperp = None

        res = GSVDResult(A, L, Uhat, Vhat, X, Y, c, s,
                         r_A, r_L, n_A, n_L,
                         Uperp=Uperp, Vperp=Vperp)
        # Attach workspace and parameters so append_column can reuse them
        res._workspace = self
        res._tol = tol
        res._full_matrices = full_matrices
        return res


def gsvd(A, L, full_matrices=False, tol=1e-12):
    """Compute the generalized SVD of (A, L) and return a GSVDResult.

    This implementation requires that the stacked matrix [A; L] has
    full column rank (up to the tolerance `tol`); otherwise a
    LinAlgError is raised.
    """
    ws = _GSVDWorkspace(A, L, full_matrices=full_matrices, tol=tol)
    return ws.to_gsvd_result()
