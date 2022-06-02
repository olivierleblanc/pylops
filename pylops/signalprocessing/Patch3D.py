import logging

import numpy as np

from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.LinearOperator import aslinearoperator
from pylops.signalprocessing.Sliding2D import _slidingsteps
from pylops.utils.tapers import tapernd

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def Patch3D(
    Op,
    dims,
    dimsd,
    nwin,
    nover,
    nop,
    tapertype="hanning",
    scalings=None,
    design=False,
    name="P",
):
    """3D Patch transform operator.

    Apply a transform operator ``Op`` repeatedly to patches of the model
    vector in forward mode and patches of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into
    patches, each patch is transformed, and patches are then recombined
    together. Both model and data are internally reshaped and
    interpreted as 3-dimensional arrays: each patch contains a portion
    of the array in every axis.

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFTND`
    or :obj:`pylops.signalprocessing.Radon3D`) on 3-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to use ``design=True`` if unsure about the
       choice ``dims`` and use the number of windows printed on screen to
       define such input parameter.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, patches may not cover the entire size of the data.
       The start and end indices of each window can be displayed using
       ``design=True`` while defining the best patching approach.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dims : :obj:`tuple`
        Shape of 3-dimensional model. Note that ``dims[0]``, ``dims[1]``
        and ``dims[2]`` should be multiple of the model size of the
        transform in their respective dimensions
    dimsd : :obj:`tuple`
        Shape of 3-dimensional data
    nwin : :obj:`tuple`
        Number of samples of window
    nover : :obj:`tuple`
        Number of samples of overlapping part of window
    nop : :obj:`tuple`
        Size of model in the transformed domain
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)
    scalings : :obj:`tuple` or :obj:`list`, optional
         Set of scalings to apply to each patch. If ``None``, no scale will be
         applied
    design : :obj:`bool`, optional
        Print number of sliding window (``True``) or not (``False``)
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    Sop : :obj:`pylops.LinearOperator`
        Sliding operator

    Raises
    ------
    ValueError
        Identified number of windows is not consistent with provided model
        shape (``dims``).

    See Also
    --------
    Patch2D: 2D Patching transform operator.

    """
    # model windows
    mwin0_ins, mwin0_ends = _slidingsteps(dims[0], nop[0], 0)
    mwin1_ins, mwin1_ends = _slidingsteps(dims[1], nop[1], 0)
    mwin2_ins, mwin2_ends = _slidingsteps(dims[2], nop[2], 0)

    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    dwin2_ins, dwin2_ends = _slidingsteps(dimsd[2], nwin[2], nover[2])
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins2 = len(dwin2_ins)
    nwins = nwins0 * nwins1 * nwins2

    # create tapers
    if tapertype is not None:
        tap = tapernd(nwin, nover, tapertype=tapertype).astype(Op.dtype)
        taps = {itap: tap for itap in range(nwins)}
        ## sides
        # topmost tapers
        taptop = tap.copy()
        taptop[: nover[0]] = tap[nwin[0] // 2]
        for itap in range(0, nwins1 * nwins2):
            taps[itap] = taptop
        # bottommost tapers
        tapbottom = tap.copy()
        tapbottom[-nover[0] :] = tap[nwin[0] // 2]
        for itap in range(nwins - nwins1 * nwins2, nwins):
            taps[itap] = tapbottom
        # frontmost tapers
        tapfront = tap.copy()
        tapfront[:, :, : nover[2]] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        for itap in range(0, nwins, nwins2):
            taps[itap] = tapfront
        # backmost tapers
        tapback = tap.copy()
        tapback[:, :, -nover[2] :] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        for itap in range(nwins2 - 1, nwins, nwins2):
            taps[itap] = tapback
        # leftmost tapers
        tapleft = tap.copy()
        tapleft[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        for itap in range(0, nwins, nwins1 * nwins2):
            for i in range(nwins2):
                taps[itap + i] = tapleft
        # rightmost tapers
        tapright = tap.copy()
        tapright[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        for itap in range(nwins2 * (nwins1 - 1), nwins, nwins2 * nwins1):
            for i in range(nwins2):
                taps[itap + i] = tapright
        ## pillars
        # topleftmost tapers
        taplefttop = tap.copy()
        taplefttop[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        taplefttop[: nover[0]] = taplefttop[nwin[0] // 2]
        for itap in range(nwins2):
            taps[itap] = taplefttop
        # toprightmost tapers
        taprighttop = tap.copy()
        taprighttop[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        taprighttop[: nover[0]] = taprighttop[nwin[0] // 2]
        for itap in range(nwins2 * (nwins1 - 1), nwins2 * nwins1):
            taps[itap] = taprighttop
        # topfrontmost tapers
        tapfronttop = tap.copy()
        tapfronttop[:, :, : nover[2]] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        tapfronttop[: nover[0]] = tapfronttop[nwin[0] // 2]
        for itap in range(0, nwins1 * nwins2, nwins2):
            taps[itap] = tapfronttop
        # topbackmost tapers
        tapbacktop = tap.copy()
        tapbacktop[:, :, -nover[2] :] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        tapbacktop[: nover[0]] = tapbacktop[nwin[0] // 2]
        for itap in range(nwins2 - 1, nwins1 * nwins2, nwins2):
            taps[itap] = tapbacktop
        """
        ## corners
        # lefttopcorner taper
        taplefttop = tap.copy()
        taplefttop[: nover[0]] = tap[nwin[0] // 2]
        taplefttop[:, : nover[1]] = taplefttop[:, nwin[1] // 2][:, np.newaxis, :]
        taplefttop[:, :, : nover[2]] = taplefttop[:, :, nwin[2] // 2][:, :, np.newaxis]
        taps[0] = taplefttop
        """
    # check that identified number of windows agrees with mode size
    if design:
        logging.warning("%d-%d-%d windows required...", nwins0, nwins1, nwins2)
        logging.warning(
            "model wins - start:%s, end:%s / start:%s, end:%s / start:%s, end:%s",
            mwin0_ins,
            mwin0_ends,
            mwin1_ins,
            mwin1_ends,
            mwin2_ins,
            mwin2_ends,
        )
        logging.warning(
            "data wins - start:%s, end:%s / start:%s, end:%s / start:%s, end:%s",
            dwin0_ins,
            dwin0_ends,
            dwin1_ins,
            dwin1_ends,
            dwin2_ins,
            dwin2_ends,
        )
    if (
        nwins0 * nop[0] != dims[0]
        or nwins1 * nop[1] != dims[1]
        or nwins2 * nop[2] != dims[2]
    ):
        raise ValueError(
            f"Model shape (dims={dims}) is not consistent with chosen "
            f"number of windows. Choose dims[0]={nwins0 * nop[0]}, "
            f"dims[1]={nwins1 * nop[1]}, and dims[2]={nwins2* nop[2]} "
            f"for the operator to work with "
            "estimated number of windows, or create "
            "the operator with design=True to find out the"
            "optimal number of windows for the current "
            "model size..."
        )

    # define scalings
    if scalings is None:
        scalings = [1.0] * nwins

    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([scalings[itap] * Op for itap in range(nwins)])
    else:
        OOp = BlockDiag(
            [
                scalings[itap] * Diagonal(taps[itap].ravel(), dtype=Op.dtype) * Op
                for itap in range(nwins)
            ]
        )

    hstack2 = HStack(
        [
            Restriction(
                (nwin[0], nwin[1], dimsd[2]),
                range(win_in, win_end),
                axis=2,
                dtype=Op.dtype,
            ).H
            for win_in, win_end in zip(dwin2_ins, dwin2_ends)
        ]
    )
    combining2 = BlockDiag([hstack2] * (nwins1 * nwins0))

    hstack1 = HStack(
        [
            Restriction(
                (nwin[0], dimsd[1], dimsd[2]),
                range(win_in, win_end),
                axis=1,
                dtype=Op.dtype,
            ).H
            for win_in, win_end in zip(dwin1_ins, dwin1_ends)
        ]
    )
    combining1 = BlockDiag([hstack1] * nwins0)

    combining0 = HStack(
        [
            Restriction(dimsd, range(win_in, win_end), axis=0, dtype=Op.dtype).H
            for win_in, win_end in zip(dwin0_ins, dwin0_ends)
        ]
    )

    Pop = aslinearoperator(combining0 * combining1 * combining2 * OOp)
    Pop.dims, Pop.dimsd = (
        nwins0,
        nwins1,
        nwins2,
        int(dims[0] // nwins0),
        int(dims[1] // nwins1),
        int(dims[2] // nwins2),
    ), dimsd

    Pop.name = name
    return Pop