"""Optics module."""

import numpy as _np

from .. import tracking as _tracking
from ..utils import interactive as _interactive


class OpticsException(Exception):
    """."""


@_interactive
def get_rf_frequency(accelerator):
    """Return the frequency of the first RF cavity in the lattice."""
    for e in accelerator:
        if e.frequency != 0.0:
            return e.frequency
    raise OpticsException('no cavity element in the lattice')


@_interactive
def get_rf_voltage(accelerator):
    """Return the voltage of the first RF cavity in the lattice."""
    voltages = []
    for e in accelerator:
        if e.voltage != 0.0:
            voltages.append(e.voltage)
    if voltages:
        if len(voltages) == 1:
            return voltages[0]
        else:
            return voltages
    else:
        raise OpticsException('no cavity element in the lattice')


@_interactive
def get_revolution_frequency(accelerator):
    """."""
    return get_rf_frequency(accelerator) / accelerator.harmonic_number


@_interactive
def get_revolution_period(accelerator):
    """."""
    return 1 / get_revolution_frequency(accelerator)


@_interactive
def get_frac_tunes(
        accelerator=None, m1turn=None, dim='6D', fixed_point=None,
        energy_offset=0.0, return_damping=False):
    """Return fractional tunes of the accelerator."""
    if m1turn is None:
        if dim == '4D':
            m1turn = _tracking.find_m44(
                accelerator, indices='m44', energy_offset=energy_offset,
                fixed_point=fixed_point)
        elif dim == '6D':
            m1turn = _tracking.find_m66(
                accelerator, indices='m66', fixed_point=fixed_point)
        else:
            raise Exception('Set valid dimension: 4D or 6D')

    traces = []
    traces.append(m1turn[0, 0] + m1turn[1, 1])
    traces.append(m1turn[2, 2] + m1turn[3, 3])
    if dim == '6D':
        traces.append(m1turn[4, 4] + m1turn[5, 5])

    evals, _ = _np.linalg.eig(m1turn)
    trc = (evals[::2] + evals[1::2]).real
    dff = (evals[::2] - evals[1::2]).imag

    trace = []
    diff = []
    for tr in traces:
        idx = _np.argmin(_np.abs(trc-tr))
        trace.append(trc[idx])
        diff.append(dff[idx])
        trc = _np.delete(trc, idx)
        dff = _np.delete(dff, idx)
    trc = _np.array(trace)
    dff = _np.array(diff)

    mus = _np.arctan2(dff, trc)
    tunes = mus / 2 / _np.pi
    if dim == '6D' and return_damping:
        alphas = trc / _np.cos(mus) / 2
        alphas = -_np.log(alphas) * get_revolution_frequency(accelerator)
        return tunes, alphas
    else:
        return tunes


@_interactive
def get_chromaticities(accelerator, energy_offset=1e-6):
    """."""
    cav_on = accelerator.cavity_on
    rad_on = accelerator.radiation_on
    accelerator.radiation_on = 0
    accelerator.cavity_on = False

    nux, nuy, *_ = get_frac_tunes(accelerator, dim='4D', energy_offset=0.0)
    nux_den, nuy_den, *_ = get_frac_tunes(
        accelerator, dim='4D', energy_offset=energy_offset)
    chromx = (nux_den - nux)/energy_offset
    chromy = (nuy_den - nuy)/energy_offset

    accelerator.cavity_on = cav_on
    accelerator.radiation_on = rad_on
    return chromx, chromy


@_interactive
def get_mcf(accelerator, order=1, energy_offset=None, energy_range=None):
    """Return momentum compaction factor of the accelerator."""
    if energy_range is None:
        energy_range = _np.linspace(-1e-3, 1e-3, 11)

    if energy_offset is not None:
        energy_range += energy_offset

    accel = accelerator[:]
    _tracking.set_4d_tracking(accel)
    leng = accel.length

    dl = _np.zeros(_np.size(energy_range))
    for i, ene in enumerate(energy_range):
        cod = _tracking.find_orbit4(accel, ene)
        cod = _np.concatenate([cod.flatten(), [ene, 0]])
        T, *_ = _tracking.ring_pass(accel, cod)
        dl[i] = T[5]/leng

    polynom = _np.polynomial.polynomial.polyfit(energy_range, dl, order)
    polynom = polynom[1:]
    if len(polynom) == 1:
        polynom = polynom[0]
    return polynom


@_interactive
def get_curlyh(beta, alpha, x, xl):
    """."""
    gamma = (1 + alpha*alpha) / beta
    return beta*xl*xl + 2*alpha*x*xl + gamma*x*x
