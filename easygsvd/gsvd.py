import numpy as np
from scipy.sparse.linalg import aslinearoperator
import scipy.sparse as sps
from scipy.linalg import null_space



class GSVDResult:
    """Provides interface to the GSVD decomposition.
    """

    def __init__(self, A, L, Uhat, Vhat, X, Y, c, s, r_A, r_L, n_A, n_L, Uperp=None, Vperp=None):
        
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

        # Partition some quantities 
        self.U1 = self.Uhat[:, :n_L]
        self.U2 = self.Uhat[:,n_L:]
        self.V2 = self.Vhat[:,:r_A-n_L]
        self.V3 =  self.Vhat[:,r_A-n_L:]
        self.X1 = self.X[:,:n_L]
        self.X2 = self.X[:,n_L:r_A]
        self.X3 = self.X[:,r_A:]
        self.Y1 = self.Y[:,:n_L]
        self.Y2 = self.Y[:,n_L:r_A]
        self.Y3 = self.Y[:,r_A:]
        self.c_hat = self.c[:r_A]
        self.s_hat = self.s[n_L:]
        self.c_check = self.c[n_L:r_A]
        self.s_check = self.s[n_L:r_A]

        # Handle full decomposition
        self._Uperp = Uperp
        self._Vperp = Vperp

        # Define full U and V if U3 and V1 provided
        self.complete = True
        if self._Uperp is not None:
            self._U = np.hstack([self.Uhat, self.Uperp])
        else:
            self.complete = False
        if self._Vperp is not None:
            self._V = np.hstack([self.Vhat, self.Vperp])
        else:
            self.complete = False

        # Generalized svals
        self.gamma_check = self.c_check / self.s_check
        self.gamma = np.hstack([ np.inf*np.ones(self.n_L), self.gamma_check, np.zeros(self.r_A) ])


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

    
    def get_orthogonal_projector(self, subspace, matrix=True):
        """Returns the orthogonal projector onto the specified subspace.
        """
        valid_subspaces = ["col(A)", "col(A.T)", "ker(A)", "ker(A.T)", 
                           "col(L)", "col(L.T)", "ker(L)", "ker(L.T)"]
        assert subspace in valid_subspaces, f"Invalid subspace, must be one of: {valid_subspaces}"

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
                return np.eye(self.Uhat.shape[0]) - ( self.Uhat @ self.Uhat.T )
            else:
                return aslinearoperator(sps.diags(np.ones(self.Uhat.shape[0]))) - ( aslinearoperator(self.Uhat) @ aslinearoperator(self.Uhat.T) )
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
                return np.eye(self.Vhat.shape[0]) - ( self.Vhat @ self.Vhat.T )
            else:
                return aslinearoperator(sps.diags(np.ones(self.Vhat.shape[0]))) - ( aslinearoperator(self.Vhat) @ aslinearoperator(self.Vhat.T) )
        else:
            raise NotImplementedError
        

    def get_oblique_projector(self, which=1, matrix=True):
        """Returns the oblique projector related to the two subspaces.
        """
        valid_options = [
            1, # projection onto ker(L) along ker(L)^{perp_A}
            2, # projection onto ker(L)^{perp_A} along ker(L)
            3, # projection onto ker(A) along ker(A)^{perp_L}
            4, # projection onto ker(A)^{perp_L} along ker(A)
        ]
        assert which in valid_options, "Invalid option, must be one of {valid_options}"

        if which == 1:
            if matrix:
                return self.X1 @ self.Y1.T
            else:
                return aslinearoperator(self.X1) @ aslinearoperator(self.Y1.T)
        elif which == 2:
            if matrix:
                return (self.X2 @ self.Y2.T) + (self.X3 @ self.Y3.T)
            else:
                return ( aslinearoperator(self.X2) @ aslinearoperator(self.Y2.T) ) + ( aslinearoperator(self.X3) @ aslinearoperator(self.Y3.T) )
        elif which == 3:
            if matrix:
                return self.X3 @ self.Y3.T
            else:
                return aslinearoperator(self.X3) @ aslinearoperator(self.Y3.T)
        elif which == 4:
            if matrix:
                return (self.X1 @ self.Y1.T) + (self.X2 @ self.Y2.T)
            else:
                return ( aslinearoperator(self.X1) @ aslinearoperator(self.Y1.T) ) + ( aslinearoperator(self.X2) @ aslinearoperator(self.Y2.T) )
        else:
            raise NotImplemented


    def get_L_oblique_pinv(self, matrix=True):
        """Returns the oblique pseudoinverse L_A^\dagger. 
        """
        if matrix:
            Lopinv = ( self.X2 @ (np.diag(1.0/self.s_check) @ self.V2.T ) ) + (self.X3 @ self.V3.T)
        else:
            Lopinv = ( aslinearoperator(self.X2) @ aslinearoperator(sps.diags(1.0/self.s_check)) @ aslinearoperator(self.V2.T) ) + ( aslinearoperator(self.X3) @ aslinearoperator(self.V3.T) ) 

        return Lopinv
    

    def get_A_oblique_pinv(self, matrix=True):
        """Returns the oblique pseudoinverse A_L^\dagger. 
        """
        if matrix:
            Lopinv = ( self.X2 @ (np.diag(1.0/self.c_check) @ self.U2.T ) ) + (self.X1 @ self.U1.T)
        else:
            Lopinv = ( aslinearoperator(self.X2) @ aslinearoperator(sps.diags(1.0/self.c_check)) @ aslinearoperator(self.U2.T) ) + ( aslinearoperator(self.X1) @ aslinearoperator(self.U1.T) ) 

        return Lopinv
    

    def get_L_standard_form_data(self, matrix=True):
        """Compute and return some quantities related to the oblique pseudoinverse.
        """
        if matrix:
            E = ( np.eye(self.X1.shape[0]) - self.X1 @ ( self.U1.T @ self.A) )
            Lopinv = self.get_L_oblique_pinv(matrix=True)
            ALopinv = self.U2 @ ( np.diag( self.gamma_check ) @ self.V2.T ) 
            kermat = self.X1 @ self.U1.T
        else:
            E = aslinearoperator(sps.diags(np.ones(self.X1.shape[0]))) - ( aslinearoperator(self.X1) @ ( aslinearoperator( ( self.A.T @ self.U1 ).T )  ) )
            Lopinv = self.get_L_oblique_pinv(matrix=False)
            ALopinv = aslinearoperator(self.U2) @ aslinearoperator( sps.diags(self.gamma_check) ) @ aslinearoperator( self.V2.T )
            kermat = aslinearoperator(self.X1) @ aslinearoperator(self.U1.T)

        return E, Lopinv, ALopinv, kermat





def gsvd(A, L, full_matrices=False, tol=1e-12):
    """Compute the generalized singular value decomposition (GSVD) of matrices A and L.
    """
    N = A.shape[1]
    assert N == L.shape[1], "A and L must have the same number of columns!"
    M = A.shape[0]
    K = L.shape[0]
    stack_matrix = np.vstack([A, L])
    assert np.linalg.matrix_rank( stack_matrix ) == N, "Common kernel condition violated!"

    # Thin QR factor the stacked matrix
    Q, R = np.linalg.qr(stack_matrix, mode='reduced')

    # Extract Q_A and Q_L
    Q_A = Q[:M, :]  # First M rows
    Q_L = Q[M:, :]  # Last K rows

    # Eigendecomposition of Q_A
    c_sq, W = np.linalg.eigh(Q_A.T @ Q_A)
    c_sq = np.clip(c_sq, a_min=0.0, a_max=None)
    c_sq = c_sq[::-1] # ascending -> descending
    W = W[:, ::-1] # ascending -> descending
    s_sq = np.diag( W.T @ ( Q_L.T @ (Q_L @ W) ) )
    c = np.sqrt(c_sq)
    s = np.sqrt(s_sq)

    # Determine nullity and rank of A and L
    computed_n_A = np.sum( c_sq < tol )
    computed_r_A = N - computed_n_A

    computed_n_L = np.sum(s_sq < tol )
    computed_r_L = N - computed_n_L

    # Hatted c and s
    c_hat = c[:computed_r_A]
    s_hat = s[computed_n_L:]
   
    # Define X
    X = np.linalg.solve(R, W)

    # Define submatrices of W
    W_A_1 = W[:,:computed_r_A]
    W_A_2 = W[:,computed_r_A:]
    W_L_1 = W[:,:computed_n_L]
    W_L_2 = W[:,computed_n_L:]

    # Define U and V
    Uhat = Q_A @ ( W_A_1 @ np.diag( 1.0/c_hat  ) )
    Vhat = Q_L @ ( W_L_2 @ np.diag( 1.0/s_hat ) )

    # Define Y
    Y = np.linalg.solve(X.T, np.eye(X.shape[0]))

    # Complete the basis for U and V?
    if full_matrices:
        Uperp = null_space(Uhat.T)
        Vperp = null_space(Vhat.T) 
    else:
        Uperp = None
        Vperp = None

    # Package into GSVDResult object
    result = GSVDResult(A, L, Uhat, Vhat, X, Y, c, s, computed_r_A, computed_r_L, computed_n_A, computed_n_L, Uperp=Uperp, Vperp=Vperp)

    return result


