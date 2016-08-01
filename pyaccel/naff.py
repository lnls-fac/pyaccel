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

import numpy as _numpy
import trackcpp as _trackcpp
import pyaccel.accelerator as _accelerator
import pyaccel.utils as _utils


_interactive = _utils.interactive

class NaffException(Exception): pass

@_interactive
def naff_traj(particles):
    """ Calculate tunes from tracking results."""
    return NotImplemented

@_interactive
def naff_general(R, I=None, nr_ff=2, use_win=1):
    """ CAlculate the first nr_ff fundamental frequencies from real (R) and (I) imaginary parts of signal"""

    if I is None: I = 0*R;

    if (len(R)-1) % 6: raise NaffException('Number of points minus 1 must be divisible by 6.')
    if not len(R) == len(I) : raise NaffException('Size of vectors R and I must be the same.')

    ff = _trackcpp.CppDoubleVector(nr_ff,0.0)
    Re = _trackcpp.CppDoubleVector(nr_ff,0.0)
    Im = _trackcpp.CppDoubleVector(nr_ff,0.0)
    _trackcpp.naff_general(R,I,nr_ff,use_win,ff,Re,Im)
    freq = _numpy.zeros(ff.size(),dtype=float)
    Four = _numpy.zeros(ff.size(),dtype=complex)
    for i in range(ff.size()):
        freq[i] = ff[i]
        Four[i] = Re[i] + 1j*Im[i]

    return freq, Four
