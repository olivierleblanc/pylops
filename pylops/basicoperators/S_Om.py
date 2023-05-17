import logging

import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

class S_Om(LinearOperator):
    r"""Restriction to set Omega (Om).

    Multiply a matrix by the left with vector :math:`\mathbf{a}`
    and by the right with vector :math:`\mathbf{b}` (:math:`\mathbf{a}` if
    :math:`\mathbf{b}` is None).

    Parameters
    ----------
    objshape : :obj:`tuple`
        Shape of the inital object.
    Om_x : :obj:`numpy.ndarray` 
        x-axis indices of set Omega.
    Om_y : :obj:`numpy.ndarray` 
        y-axis indices of set Omega.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    complex : :obj:`bool`
        Data has complex numbers (``True``) or not (``False``)

    """

    def __init__(self, objshape, Om_x, Om_y, mult_mat=None, dtype=np.complex128):
        self.objshape = objshape
        self.Q = Om_x.shape[0]
        self.Om_x = Om_x.reshape(-1)
        self.Om_y = Om_y.reshape(-1)
        self.mult_mat = mult_mat
        self.shape = (len(self.Om_x), np.prod(self.objshape))
        self.dtype = np.dtype(dtype)
       
    def _matvec(self, X):
        """Subsample 2D signal 'X' at indices 'Om_x' and 'Om_y'.
        The multiplicity of each index is corrected.

        Args:
            X (2D array)        : Input signal.
            Om_x (1D array)     : core position pairwise difference along x-axis
            Om_y (1D array)     : core position pairwise difference along y-axis
            mult_mat (2D array) : matrix containing the multiplicity of each Fourier coeff

        Returns:
            S_om_X (2D array): X[Om_x,Om_y].

        """
        X = X.reshape(self.objshape)
        S_Om_X = X[self.Om_y, self.Om_x] 
        
        if (self.mult_mat is not None):
            S_Om_X = X[self.Om_y,self.Om_x]/np.sqrt(self.mult_mat)

        return S_Om_X

    def _rmatvec(self, U):
        """Adjoint of S_Om. Fill a 2D array at indices 'Om_x' and 'Om_y' with 'U'.
        Note: regarding the notes, S_Om_star not compact must consider the multiplicities,
                they are thus added to keep the non compact format. 
        The 'diag(w)' term is finally corrected with diag(1/sqrt(w)) in both S_Om and S_Om_star

        Args:
            U (2D array)  : Input signal.

        Returns:
            S_om_star_U (2D array).

        """
        S_Om_star_U = np.zeros(self.objshape, dtype=complex)

        if (self.mult_mat is None):
            np.add.at(S_Om_star_U, (self.Om_y, self.Om_x), U) 
        else :
            np.add.at(S_Om_star_U, (self.Om_y, self.Om_x), U/np.sqrt(self.mult_mat)) 
                
        return S_Om_star_U