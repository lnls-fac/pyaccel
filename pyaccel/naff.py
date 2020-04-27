""" Pyaccel tracking module

This module concentrates all tracking routines of the accelerator.
Most of them take a structure called 'positions' as an argument which
should store the initial coordinates of the particle, or the bunch of particles
to be tracked. Most of these routines generate tracked particle positions as
output, among other data. These input and ouput particle positions are stored
as native python lists or numpy arrays. Depending on the routine these position
objects may have one, two, three or four indices. These indices are used to
select particle's inices (p), coordinates(c), lattice element indices(e) or
turn number (n). For example, v = pos[p,c,e,n]. Routines in these module may
return particle positions structure missing one or more indices but the
PCEN ordering is preserved.
"""

import numpy as _np
import trackcpp as _trackcpp
import pyaccel.utils as _utils


_interactive = _utils.interactive


class NaffException(Exception):
    pass


@_interactive
def naff_general(signal, is_real=True, nr_ff=2, window=1):
    """Calculate the first `nr_ff` fundamental frequencies of `signal`.

    Inputs:
        signal -- 1D or 2D Numpy array. In case of 2D Numpy array NAFF will
            be applied to each row of `signal`;
        is_real -- Whether to consider `signal` as real;
        nr_ff -- Number of fundamental frequencies to return (default = 2);
        window -- Which window to use:  (default = 1)
            0 -- no window;
            1, 2, 3, ... -- Powers of hanning window;
            -1 -- Exponential window;

    Outputs:
        freqs -- fundamental frequencies.
            Numpy array with shape `(signal.shape[0], nr_ff)`.
            In case is_real is true, only positive frequencies in the interval
            [0, 0.5] are returned;
        fourier -- Fourier component of the given frequencies.
            Numpy array of complex numbers with same shape as freqs.
    """
    if signal.ndim == 1:
        signal = signal[None, :]

    if signal.ndim > 2:
        NaffException('Wrong number of dimensions for input array.')

    if (signal.shape[1]-1) % 6:
        raise NaffException('Number of points minus 1 must be divisible by 6.')

    freqs = _np.zeros((signal.shape[0], nr_ff))
    real = _np.zeros((signal.shape[0], nr_ff))
    imag = _np.zeros((signal.shape[0], nr_ff))
    _trackcpp.naff_general_wrapper(
        signal.real, signal.imag, is_real, nr_ff, window, freqs, real, imag)

    fourier = real + 1j*imag
    fourier = _np.squeeze(fourier)
    freqs = _np.squeeze(freqs)
    return freqs, fourier
