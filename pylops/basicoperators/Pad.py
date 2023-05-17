import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple


class Pad(LinearOperator):
    r"""Pad operator.

    Zero-pad model in forward model and extract non-zero subsequence
    in adjoint. Padding can be performed in one or multiple directions to any
    multi-dimensional input arrays.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension
    pad : :obj:`tuple`
        Number of samples to pad. If ``dims`` is a scalar, ``pad`` is a single
        tuple ``(pad_in, pad_end)``. If ``dims`` is a tuple,
        ``pad`` is a tuple of tuples where each inner tuple contains
        the number of samples to pad in each dimension
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    ValueError
        If any element of ``pad`` is negative.

    Notes
    -----
    Given an array of size :math:`N`, the *Pad* operator simply adds
    :math:`\text{pad}_\text{in}` at the start and :math:`\text{pad}_\text{end}` at the end in forward mode:

    .. math::

        y_{i} = x_{i-\text{pad}_\text{in}}  \quad \forall
        i=\text{pad}_\text{in},\ldots,\text{pad}_\text{in}+N-1

    and :math:`y_i = 0 \quad \forall
    i=0,\ldots,\text{pad}_\text{in}-1, \text{pad}_\text{in}+N-1,\ldots,N+\text{pad}_\text{in}+\text{pad}_\text{end}`

    In adjoint mode, values from :math:`\text{pad}_\text{in}` to :math:`N-\text{pad}_\text{end}` are
    extracted from the data:

    .. math::

        x_{i} = y_{\text{pad}_\text{in}+i}  \quad \forall i=0, N-1

    """

    def __init__(self, dims, pad, dtype="float64", name="P"):
        if np.any(np.array(pad) < 0):
            raise ValueError("Padding must be positive or zero")
        self.reshape = False if isinstance(dims, int) else True
        self.dims = _value_or_list_like_to_tuple(dims)
        self.pad = pad
        if self.reshape:
            dimsd = [
                dim + before + after
                for dim, (before, after) in zip(self.dims, self.pad)
            ]
        else:
            dimsd = [self.dims[0] + pad[0] + pad[1]]
        self.dimsd = tuple(dimsd)

        self.shape = (np.prod(self.dimsd), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        super().__init__(explicit=False, clinear=True, name=name)

    def _matvec(self, x):
        if self.reshape:
            y = x.reshape(self.dims)
            y = np.pad(y, self.pad, mode="constant")
        else:
            y = np.pad(x, self.pad, mode="constant")
        return y.ravel()

    def _rmatvec(self, x):
        if self.reshape:
            y = x.reshape(self.dimsd)
            for ax, (before, _) in enumerate(self.pad):
                y = np.take(y, np.arange(before, before + self.dims[ax]), axis=ax)
        else:
            y = x[self.pad[0] : self.pad[0] + self.dims[0]]
        return y.ravel()