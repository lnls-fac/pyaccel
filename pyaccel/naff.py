"""NAFF module
This module performs the Numerical Analysis of Fundamental Frequencies (NAFF)
method, developed by J. Naskar in:

J. Naskar, The chaotic motion of the solar system: A numerical estimate of the
size of the chaotic zones, Icarus, Volume 88, Issue 2,1990, Pages 266-291.
(https://www.sciencedirect.com/science/article/pii/001910359090084M)
"""

import numpy as _np

import trackcpp as _trackcpp

from .utils import interactive as _interactive


class NaffException(Exception):
    """."""


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
            1, 2, 3, ... -- Powers of Hanning window;
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
        q, r = divmod(signal.shape[1], 6)
        if r == 0:
            q -= 1
        q = 6*q+1
        signal = signal[:, :q]

    freqs = _np.zeros((signal.shape[0], nr_ff))
    real = _np.zeros((signal.shape[0], nr_ff))
    imag = _np.zeros((signal.shape[0], nr_ff))
    _trackcpp.naff_general_wrapper(
        signal.real, signal.imag, is_real, nr_ff, window, freqs, real, imag)

    fourier = real + 1j*imag
    fourier = _np.squeeze(fourier)
    freqs = _np.squeeze(freqs)

    if is_real:
        freqs = _np.abs(freqs)

    return freqs, fourier
