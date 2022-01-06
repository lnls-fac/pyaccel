"""."""

import numpy as _np

from .. import tracking as _tracking
from .. import lattice as _lattice
from ..utils import interactive as _interactive

from .twiss import calc_twiss as _calc_twiss
from .miscellaneous import OpticsException as _OpticsException, \
    get_curlyh as _get_curlyh


@_interactive
def calc_transverse_acceptance(
        accelerator, twiss=None, init_twiss=None, fixed_point=None,
        energy_offset=0.0):
    """Return transverse horizontal and vertical physical acceptances."""
    if twiss is None:
        twiss, _ = _calc_twiss(
            accelerator, init_twiss=init_twiss, fixed_point=fixed_point,
            indices='closed', energy_offset=energy_offset)

    n_twi = len(twiss)
    n_acc = len(accelerator)
    if n_twi not in {n_acc, n_acc+1}:
        raise _OpticsException(
            'Mismatch between size of accelerator and size of twiss object')

    betax = twiss.betax
    betay = twiss.betay
    co_x = twiss.rx
    co_y = twiss.ry

    # physical apertures
    hmax, hmin = _np.zeros(n_twi), _np.zeros(n_twi)
    vmax, vmin = _np.zeros(n_twi), _np.zeros(n_twi)
    for idx, ele in enumerate(accelerator):
        hmax[idx], hmin[idx] = ele.hmax, ele.hmin
        vmax[idx], vmin[idx] = ele.vmax, ele.vmin
    if n_twi > n_acc:
        hmax[-1], hmin[-1] = hmax[0], hmin[0]
        vmax[-1], vmin[-1] = vmax[0], vmin[0]

    # calcs acceptance with beta at entrance of elements
    betax_sqrt, betay_sqrt = _np.sqrt(betax), _np.sqrt(betay)
    accepx_pos = (hmax - co_x)
    accepx_neg = (-hmin + co_x)
    accepy_pos = (vmax - co_y)
    accepy_neg = (-vmin + co_y)
    accepx_pos /= betax_sqrt
    accepx_neg /= betax_sqrt
    accepy_pos /= betay_sqrt
    accepy_neg /= betay_sqrt
    accepx = _np.minimum(accepx_pos, accepx_neg)
    accepy = _np.minimum(accepy_pos, accepy_neg)
    accepx[accepx < 0] = 0
    accepy[accepy < 0] = 0
    accepx *= accepx
    accepy *= accepy
    return accepx, accepy, twiss


@_interactive
def calc_touschek_energy_acceptance(
        accelerator, energy_offsets=None, track=False, check_tune=False,
        **kwargs):
    """."""
    vcham_sts = accelerator.vchamber_on
    rad_sts = accelerator.radiation_on
    cav_sts = accelerator.cavity_on

    accelerator.radiation_on = False
    accelerator.cavity_on = False
    accelerator.vchamber_on = False

    if energy_offsets is None:
        energy_offsets = _np.linspace(1e-6, 6e-2, 60)

    if _np.any(energy_offsets < 0):
        raise ValueError('delta must be a positive vector.')

    # ############ Calculate physical aperture ############

    hmax = _lattice.get_attribute(accelerator, 'hmax', indices='closed')
    hmin = _lattice.get_attribute(accelerator, 'hmin', indices='closed')
    twi0, *_ = _calc_twiss(accelerator, indices='closed')
    tune0 = _np.array([twi0[-1].mux, twi0[-1].muy]) / (2*_np.pi)
    rx0 = twi0.rx
    px0 = twi0.px

    # positive energies
    curh_pos, ap_phys_pos, tune_pos, beta_pos = _calc_phys_apert_for_touschek(
        accelerator, energy_offsets, rx0, px0, hmax, hmin)
    # negative energies
    curh_neg, ap_phys_neg, tune_neg, beta_neg = _calc_phys_apert_for_touschek(
        accelerator, -energy_offsets, rx0, px0, hmax, hmin)

    # Considering synchrotron oscillations, negative energy deviations will
    # turn into positive ones and vice-versa, so the apperture must be
    # symmetric:
    ap_phys = _np.minimum(ap_phys_pos, ap_phys_neg)

    # ############ Calculate Dynamic Aperture ############
    ap_dyn_pos = _np.full(energy_offsets.shape, _np.inf)
    ap_dyn_neg = ap_dyn_pos.copy()
    if track:
        nturns = kwargs.get('track_nr_turns', 131)
        parallel = kwargs.get('track_parallel', True)
        curh_track = kwargs.get(
            'track_curh', _np.linspace(0, 4e-6, 30))
        ener_pos = kwargs.get(
            'track_delta_pos', _np.linspace(0.02, energy_offsets.max(), 20))
        ener_neg = kwargs.get('track_delta_neg', -ener_pos)

        beta_pos = _np.interp(ener_pos, energy_offsets, beta_pos)
        ap_dyn_pos = _calc_dyn_apert_for_touschek(
            accelerator, ener_pos, curh_track, beta_pos, nturns,
            parallel=parallel)
        ap_dyn_pos = _np.interp(energy_offsets, ener_pos, ap_dyn_pos)

        beta_neg = _np.interp(-ener_neg, energy_offsets, beta_neg)
        ap_dyn_neg = _calc_dyn_apert_for_touschek(
            accelerator, ener_neg, curh_track, beta_neg, nturns,
            parallel=parallel)
        ap_dyn_neg = _np.interp(energy_offsets, -ener_neg, ap_dyn_neg)

    accelerator.vchamber_on = vcham_sts
    accelerator.radiation_on = rad_sts
    accelerator.cavity_on = cav_sts

    # ############ Check tunes ############
    # Make sure tunes don't cross int and half-int resonances
    # Must be symmetric due to syncrhotron oscillations
    ap_tune = _np.full(energy_offsets.shape, _np.inf)
    if check_tune:
        tune0_int = _np.floor(tune0)
        tune_pos -= tune0_int[:, None]
        tune_neg -= tune0_int[:, None]
        tune0 -= tune0_int

        # make M*tune > b to check for quadrant crossing:
        quadrant0 = _np.array(tune0 > 0.5, dtype=float)
        M = _np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        b = _np.array([0, 0, -1/2, -1/2])
        b += _np.r_[quadrant0/2, -quadrant0/2]

        boo_pos = _np.dot(M, tune_pos) > b[:, None]
        boo_neg = _np.dot(M, tune_neg) > b[:, None]
        nok_pos = ~boo_pos.all(axis=0)
        nok_neg = ~boo_neg.all(axis=0)

        ap_tune[nok_pos] = 0
        ap_tune[nok_neg] = 0

    # ############ Calculate aperture ############
    ap_pos = _np.minimum(ap_dyn_pos, ap_phys, ap_tune)
    ap_neg = _np.minimum(ap_dyn_neg, ap_phys, ap_tune)
    # make sure it is monotonic with energy:
    for idx in _np.arange(1, ap_pos.size):
        ap_pos[idx] = _np.minimum(ap_pos[idx], ap_pos[idx-1])
    for idx in _np.arange(1, ap_neg.size):
        ap_neg[idx] = _np.minimum(ap_neg[idx], ap_neg[idx-1])

    # ############ Calculate energy acceptance along ring ############
    comp = curh_pos >= ap_pos[:, None]
    idcs = _np.argmax(comp, axis=0)
    boo = _np.take_along_axis(comp, _np.expand_dims(idcs, axis=0), axis=0)
    idcs[~boo.ravel()] = ap_pos.size-1
    accep_pos = energy_offsets[idcs]

    comp = curh_neg >= ap_neg[:, None]
    idcs = _np.argmax(comp, axis=0)
    boo = _np.take_along_axis(comp, _np.expand_dims(idcs, axis=0), axis=0)
    idcs[~boo.ravel()] = ap_neg.size-1
    accep_neg = -energy_offsets[idcs]

    return accep_neg, accep_pos


def _calc_phys_apert_for_touschek(
        accelerator, energy_offsets, rx0, px0, hmax, hmin):
    curh = _np.full((energy_offsets.size, rx0.size), _np.inf)
    tune = _np.full((2, energy_offsets.size), _np.nan)
    ap_phys = _np.zeros(energy_offsets.size)
    beta = _np.ones(energy_offsets.size)
    try:
        for idx, delta in enumerate(energy_offsets):
            twi, *_ = _calc_twiss(
                accelerator, energy_offset=delta, indices='closed')
            if _np.any(_np.isnan(twi[0].betax)):
                raise _OpticsException('error')
            tune[0, idx] = twi[-1].mux / (2*_np.pi)
            tune[1, idx] = twi[-1].muy / (2*_np.pi)
            beta[idx] = twi[0].betax
            rx = twi.rx
            px = twi.px
            betax = twi.betax
            dcox = rx - rx0
            dcoxp = px - px0
            curh[idx] = _get_curlyh(betax, twi.alphax, dcox, dcoxp)

            apper_loc = _np.minimum((hmax - rx)**2, (hmin + rx)**2)
            ap_phys[idx] = _np.min(apper_loc / betax)
    except (_OpticsException, _tracking.TrackingException):
        pass
    return curh, ap_phys, tune, beta


def _calc_dyn_apert_for_touschek(
        accelerator, energies, curh, beta, nturns, parallel=False):
    accelerator.cavity_on = False
    accelerator.radiation_on = False
    accelerator.vchamber_on = False

    rin = _np.full((6, energies.size, curh.size), _np.nan)
    try:
        for idx, en in enumerate(energies):
            rin[:4, idx, :] = _tracking.find_orbit4(
                accelerator, energy_offset=en).ravel()[:, None]
    except _tracking.TrackingException:
        pass
    rin = rin.reshape(6, -1)

    accelerator.cavity_on = True
    accelerator.radiation_on = True
    accelerator.vchamber_on = True
    orb6d = _tracking.find_orbit6(accelerator)

    # Track positive energies
    curh0, ener = _np.meshgrid(curh, energies)
    xl = _np.sqrt(curh0/beta[:, None])

    rin[1, :] += xl.ravel()
    rin[2, :] += 1e-6
    rin[4, :] = orb6d[4] + ener.ravel()
    rin[5, :] = orb6d[5]

    _, _, lostturn, *_ = _tracking.ring_pass(
        accelerator, rin, nturns, turn_by_turn=False, parallel=parallel)
    lostturn = _np.reshape(lostturn, curh0.shape)
    lost = lostturn != nturns

    ind_dyn = _np.argmax(lost, axis=1)
    ap_dyn = curh[ind_dyn]
    return ap_dyn
