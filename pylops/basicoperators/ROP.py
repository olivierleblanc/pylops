import logging

import numpy as np
import scipy as sp
from scipy.sparse.linalg import inv

from pylops import LinearOperator
from pylops.utils.backend import get_array_module

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

def rmdiag(X):
    """Remove the diagonal of a matrix X"""
    return X - np.diag(np.diag(X))

class ROP(LinearOperator):
    r"""Rank-One Projection (ROP).

    Multiply a matrix by the left with vector :math:`\mathbf{a}`
    and by the right with vector :math:`\mathbf{b}` (:math:`\mathbf{a}` if
    :math:`\mathbf{b}` is None).

    Parameters
    ----------
    a_ij : :obj:`numpy.ndarray` 
        Left arrays.
    b_ij : :obj:`numpy.ndarray` 
        Right arrays.
    diagless : :obj:`bool`
        Remove the diagonal or not.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    complex : :obj:`bool`
        Matrix has complex numbers (``True``) or not (``False``)

    """

    def __init__(self, a_ij, b_ij=None, diagless=True, dtype="float64"):
        ncpa = get_array_module(a_ij)
        ncpb = get_array_module(b_ij)
        self.a_ij = a_ij
        self.M, self.Q = self.a_ij.shape
        self.b_ij = b_ij
        self.diagless = diagless
        if isinstance(a_ij, ncpa.ndarray):
            self.complex = np.iscomplexobj(a_ij) 
        else:
            self.complex = np.iscomplexobj(a_ij.data)
        if (b_ij is not None):
            if isinstance(b_ij, ncpb.ndarray):
                self.complex = self.complex or np.iscomplexobj(b_ij) 
            else:
                self.complex = self.complex or np.iscomplexobj(b_ij.data)
        self.shape = (self.a_ij.shape[0], \
                self.a_ij.shape[1]* self.a_ij.shape[1])
        self.dtype = np.dtype(dtype)
        # if a_ij or b_ij is complex, check for correctness or upcast to complex
        if self.complex and not np.iscomplexobj(np.ones(1, dtype=self.dtype)):
            self.dtype=np.complex128
            logging.warning(
                "Vector a_ij or b_ij is a complex object, dtype cast to %s" % self.dtype
            )

    def _matvec(self, X):
        X = X.reshape((self.Q, self.Q))
        if (self.diagless):
            X = rmdiag(X)
        if (self.b_ij is None):
            y = np.sum( (self.a_ij.conj()*(self.a_ij@X.T)), axis=1)
        else:
            assert self.a_ij.shape==self.b_ij.shape, "a_ij and b_ij should contain the same number of elements"
            y = np.sum( (self.a_ij.conj()*(self.b_ij@X.T)), axis=1) # Written properly this is equivalent
        return y/np.sqrt(self.M)

    def _rmatvec(self, y):

        if (self.b_ij is None):
            z = self.a_ij.T @ np.diag(y) @ self.a_ij.conj()
        else:
            assert self.a_ij.shape==self.b_ij.shape, "a_ij and b_ij should contain the same number of elements"
            z = self.a_ij.T @ np.diag(y) @ self.b_ij.conj()

        if (self.diagless):
            z = rmdiag(z)
        return z/np.sqrt(self.M)