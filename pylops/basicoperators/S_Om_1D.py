import logging

import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

class S_Om_1D(LinearOperator):
    r"""Restriction to set Omega (Om).

    Multiply a matrix by the left with vector :math:`\mathbf{a}`
    and by the right with vector :math:`\mathbf{b}` (:math:`\mathbf{a}` if
    :math:`\mathbf{b}` is None).

    Parameters
    ----------
    N    : :obj:`int`
        Shape of the inital vector.
    Om : :obj:`numpy.ndarray` 
        Indices of set Omega.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    complex : :obj:`bool`
        Data has complex numbers (``True``) or not (``False``)

    """

    def __init__(self, N, Om, mult_mat=None, dtype=np.complex128):
        self.N = N
        self.Om = Om.reshape(-1)
        self.mult_mat = mult_mat
        self.shape = (len(self.Om), self.N)
        self.dtype = np.dtype(dtype)
       
    def _matvec(self, x):
        """Subsample 1D signal 'x' at indices 'Om'.
        The multiplicity of each index is corrected.

        Args:
            x (1D array) : Input signal.
            Om (2D array): The indices

        Returns:
            S_om_x (2D array): x[Om].

        """
        # X = X.reshape((self.Q, self.Q))
        S_Om_x = x[self.Om] 
        
        if (self.mult_mat is not None):
            S_Om_x = x[self.Om]/np.sqrt(self.mult_mat)

        # "Avoid aliasing"
        # ind = np.abs(Om)>=len(x)/2 
        # S_Om_x[ind] = 0

        return S_Om_x

    def _rmatvec(self, U):
        """Adjoint of S_Om. Fill a 1D array at indices 'Om' with 'u'.
        Note: regarding the notes, S_Om_star not compact must consider the multiplicities,
                they are thus added to keep the non compact format. 
        The 'diag(w)' term is finally corrected with diag(1/sqrt(w)) in both S_Om and S_Om_star

        Args:
            U (2D array)  : Input signal.

        Returns:
            S_om_star_U (1D array).

        """
        S_Om_star_U = np.zeros(self.N, dtype=complex)

        if (self.mult_mat is None):
            np.add.at(S_Om_star_U, self.Om, U) 
        else :
            np.add.at(S_Om_star_U, self.Om, U/np.sqrt(self.mult_mat)) 
                
        return S_Om_star_U