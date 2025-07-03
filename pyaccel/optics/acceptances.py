"""."""

import numpy as _np

from .. import lattice as _lattice, tracking as _tracking
from ..utils import interactive as _interactive
from .miscellaneous import get_curlyh as _get_curlyh, \
    OpticsError as _OpticsError
from .twiss import calc_twiss as _calc_twiss


@_interactive
def calc_transverse_acceptance(
    accelerator,
    twiss=None,
    init_twiss=None,
    fixed_point=None,
    energy_offset=0.0,
):
    """Return transverse horizontal and vertical physical acceptances.

    Args:
        accelerator (pyaccel.accelerator.Accelerator): accelerator object;
        twiss (numpy.ndarray, dtype=twiss, optional): Twiss object. Must have
            len(accelerator) + 1 size. If not provided, will be calculated
            internally. Defaults to None.
        init_twiss (numpy.ndarray, dtype=twiss, optional): initial twiss
            object for twiss calculation. If not provided, periodic conditions
            will be assumed. Defaults to None.
        fixed_point (np.ndarray|6-list, optional): initial point of
            tracjectory around which to calculate twiss and acceptances. If
            not provided, the fixed point of the accelerator will be used.
            Defaults to None.
        energy_offset (float, optional): Energy offset used to calculate twiss
            and fixed point fo acceptance calculations. Defaults to 0.0.

    Raises:
        _OpticsException: When user provided twiss object does not have the
            appropriate size.

    Returns:
        accepx: horizontal acceptance in units of [m.rad];
        accepy: vertical acceptance in units of [m.rad];
        twiss: twiss object.
    """
    if twiss is None:
        twiss, _ = _calc_twiss(
            accelerator,
            init_twiss=init_twiss,
            fixed_point=fixed_point,
            indices='closed',
            energy_offset=energy_offset
        )

    n_twi = len(twiss)
    n_acc = len(accelerator)
    if n_twi not in {n_acc, n_acc+1}:
        raise _OpticsError(
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
    accepx_pos = hmax - co_x
    accepx_neg = -hmin + co_x
    accepy_pos = vmax - co_y
    accepy_neg = -vmin + co_y
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
def calc_beam_stay_clear(
    accelerator,
    twiss=None,
    init_twiss=None,
    fixed_point=None,
    energy_offset=0.0
):
    """Return transverse horizontal and vertical physical acceptances.

    Args:
        accelerator (pyaccel.accelerator.Accelerator): accelerator object;
        twiss (numpy.ndarray, dtype=twiss, optional): Twiss object. Must have
            len(accelerator) + 1 size. If not provided, will be calculated
            internally. Defaults to None.
        init_twiss (numpy.ndarray, dtype=twiss, optional): initial twiss
            object for twiss calculation. If not provided, periodic conditions
            will be assumed. Defaults to None.
        fixed_point (np.ndarray|6-list, optional): initial point of
            tracjectory around which to calculate twiss and acceptances. If
            not provided, the fixed point of the accelerator will be used.
            Defaults to None.
        energy_offset (float, optional): Energy offset used to calculate twiss
            and fixed point fo acceptance calculations. Defaults to 0.0.

    Raises:
        _OpticsException: When user provided twiss object does not have the
            appropriate size.

    Returns:
        bscx: horizontal beam stay clear in units of [m];
        bscy: vertical beam stay clear in units of [m];
        twiss: twiss object.
    """
    accepx, accepy, twiss = calc_transverse_acceptance(
        accelerator, twiss, init_twiss, fixed_point, energy_offset
    )
    accx = _np.min(accepx)
    accy = _np.min(accepy)
    return _np.sqrt(accx*twiss.betax), _np.sqrt(accy*twiss.betay), twiss


@_interactive
def calc_touschek_energy_acceptance(
    accelerator,
    energy_offsets=None,
    check_tune=False,
    tune_matrix=None,
    track=False,
    track_nr_turns=131,
    track_parallel=True,
    track_curh=None,
    track_delta_pos=None,
    track_delta_neg=None,
    return_full=False,
):
    """Calculate Touschek scattering energy acceptance.

    Args:
        accelerator (pyaccel.accelerator.Accelerator): Accelerator model.
        energy_offsets (numpy.ndarray, optional): array with energy deviations
            where to calculate twiss parameters. Defaults to None.
            If None, then `numpy.linspace(1e-6, 6e-2, 60)` will be used.
        check_tune (bool, optional): Whether to check if tune crosses
            important resonances. Defaults to False.
        tune_matrix (numpy.ndarray, (n, 3), optional): Matrix describing a 2D
            convex region where tunes must lie. If for some energies the tune
            crosses the vertices of the region, the maximum/miminum positive/
            negative energy for which the tune did not cross will be
            considered as the energy acceptance. The inequalities defining the
            OK region will be interpreted this way:
                M_11 * nux + M_12 * nuy > M_13
                         ...
                M_n1 * nux + M_n2 * nuy > M_n3
            where nux and nuy are the fractional part of the tunes. If all
            inequalites are True, then the tunes will be considered OK.
            Defaults to None. If None, then the semi-integer square sorrouding
            the on-momentum tune will be used.
        track (bool, optional): Whether to perform tracking to refine
            acceptance calculation procedure. Defaults to False.
        track_nr_turns (int, optional): Number of turns to track particles. It
            is recommended to track for at least half turn on the longitudinal
            phase space. Defaults to 131.
        track_parallel (bool, optional): Whether to perform tracking in
            parallel. Defaults to True.
        track_curh (numpy.ndarray, optional): Array containing the actions to
            consider in tracking. Defaults to None. If None, then
            `numpy.linspace(0, 4e-6, 30)` will be used.
        track_delta_pos (numpy.ndarray, optional): Array containing posivite
            energy deviations to consider in tracking. Defaults to None. If
            None, then `numpy.linspace(0.02, energy_offsets.max(), 20)` will
            be used.
        track_delta_neg (numpy.ndarray, optional): Array containing negative
            energy deviations to consider in tracking. Defaults to None. If
            None, then `-track_delta_pos` will be used.
        return_full (bool, optional): Whether to return dictionary with
            intermediate results. Defaults to False.

    Raises:
        ValueError: raised if any value of `energy_offsets` is negative.

    Returns:
        _type_: _description_
    """
    vcham_sts = accelerator.vchamber_on
    rad_sts = accelerator.radiation_on
    cav_sts = accelerator.cavity_on

    accelerator.radiation_on = 'off'
    accelerator.cavity_on = False
    accelerator.vchamber_on = False

    if energy_offsets is None:
        energy_offsets = _np.linspace(1e-6, 6e-2, 60)

    dic = dict()
    if _np.any(energy_offsets < 0):
        raise ValueError('delta must be a positive vector.')
    dic['energy_offsets'] = energy_offsets

    # ############ Calculate physical aperture ############
    hmax = _lattice.get_attribute(accelerator, 'hmax', indices='closed')
    hmin = _lattice.get_attribute(accelerator, 'hmin', indices='closed')
    twi0, *_ = _calc_twiss(accelerator, indices='closed')
    tune0 = _np.array([twi0[-1].mux, twi0[-1].muy]) / (2*_np.pi)
    rx0 = twi0.rx
    px0 = twi0.px
    dic['hmax'] = hmax
    dic['hmin'] = hmin

    # positive energies
    res = _calc_phys_apert_for_touschek(
        accelerator, energy_offsets, rx0, px0, hmax, hmin
    )
    curh_pos, ap_phys_pos, tune_pos, beta_pos, idx_max_pos, twip = res
    # negative energies
    res = _calc_phys_apert_for_touschek(
        accelerator, -energy_offsets, rx0, px0, hmax, hmin
    )
    curh_neg, ap_phys_neg, tune_neg, beta_neg, idx_max_neg, twin = res
    dic['curh_pos'] = curh_pos
    dic['curh_neg'] = curh_neg
    dic['ap_phys_pos'] = ap_phys_pos
    dic['ap_phys_neg'] = ap_phys_neg
    dic['ap_phys_pos_idx'] = idx_max_pos
    dic['ap_phys_neg_idx'] = idx_max_neg
    dic['tune_pos'] = tune_pos
    dic['tune_neg'] = tune_neg
    dic['beta_pos'] = beta_pos
    dic['beta_neg'] = beta_neg
    dic['twi_pos'] = twip
    dic['twi_neg'] = twin

    # Considering synchrotron oscillations, negative energy deviations will
    # turn into positive ones and vice-versa, so the apperture must be
    # symmetric:
    ap_phys = _np.minimum(ap_phys_pos, ap_phys_neg)
    dic['ap_phys'] = ap_phys

    # ############ Calculate Dynamic Aperture ############
    ap_dyn_pos = _np.full(energy_offsets.shape, _np.inf)
    ap_dyn_neg = ap_dyn_pos.copy()
    if track:
        nturns = track_nr_turns
        parallel = track_parallel
        if track_curh is None:
            track_curh = _np.linspace(0, 4e-6, 30)
        if track_delta_pos is None:
            track_delta_pos = _np.linspace(0.02, energy_offsets.max(), 20)
        if track_delta_neg is None:
            track_delta_neg = -track_delta_pos

        beta_pos = _np.interp(track_delta_pos, energy_offsets, beta_pos)
        ap_dyn_pos = _calc_dyn_apert_for_touschek(
            accelerator,
            track_delta_pos,
            track_curh,
            beta_pos,
            nturns,
            parallel=parallel
        )
        ap_dyn_pos = _np.interp(energy_offsets, track_delta_pos, ap_dyn_pos)

        beta_neg = _np.interp(-track_delta_neg, energy_offsets, beta_neg)
        ap_dyn_neg = _calc_dyn_apert_for_touschek(
            accelerator,
            track_delta_neg,
            track_curh,
            beta_neg,
            nturns,
            parallel=parallel
        )
        ap_dyn_neg = _np.interp(energy_offsets, -track_delta_neg, ap_dyn_neg)
    dic['ap_dyn_pos'] = ap_dyn_pos
    dic['ap_dyn_neg'] = ap_dyn_neg

    accelerator.vchamber_on = vcham_sts
    accelerator.radiation_on = rad_sts
    accelerator.cavity_on = cav_sts

    # ############ Check tunes ############
    # Make sure tunes don't cross int and half-int resonances
    # Must be symmetric due to synchrotron oscillations
    ap_tune_pos = _np.full(energy_offsets.shape, _np.inf)
    ap_tune_neg = ap_tune_pos.copy()
    if check_tune:
        tune0_int = _np.floor(tune0)
        tune_pos -= tune0_int[:, None]
        tune_neg -= tune0_int[:, None]
        tune0 -= tune0_int

        # make M*tune > b to check for quadrant crossing:
        if tune_matrix is None:
            quadrant0 = _np.array(tune0 > 0.5, dtype=float)
            M = _np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
            b = _np.array([0, 0, -1/2, -1/2])
            b += _np.r_[quadrant0/2, -quadrant0/2]
        else:
            M = tune_matrix[:, :2]
            b = tune_matrix[:, -1]

        boo_pos = _np.dot(M, tune_pos) > b[:, None]
        boo_neg = _np.dot(M, tune_neg) > b[:, None]
        nok_pos = ~boo_pos.all(axis=0)
        nok_neg = ~boo_neg.all(axis=0)

        ap_tune_pos[nok_pos] = 0
        ap_tune_neg[nok_neg] = 0
    ap_tune = _np.minimum(ap_phys_pos, ap_phys_neg)
    dic['ap_tune_pos'] = ap_tune_pos
    dic['ap_tune_neg'] = ap_tune_neg
    dic['ap_tune'] = ap_tune

    # ############ Calculate aperture ############
    ap_pos = _np.minimum(ap_dyn_pos, ap_phys, ap_tune)
    ap_neg = _np.minimum(ap_dyn_neg, ap_phys, ap_tune)
    # make sure it is monotonic with energy:
    for idx in _np.arange(1, ap_pos.size):
        ap_pos[idx] = _np.minimum(ap_pos[idx], ap_pos[idx-1])
    for idx in _np.arange(1, ap_neg.size):
        ap_neg[idx] = _np.minimum(ap_neg[idx], ap_neg[idx-1])
    dic['ap_pos'] = ap_pos
    dic['ap_neg'] = ap_neg

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

    dic_r = {
        'spos': twi0.spos,
        'accp': accep_pos,
        'accn': accep_neg,
    }

    if return_full:
        dic_r.update(dic)
    return dic_r


def _calc_phys_apert_for_touschek(
    accelerator,
    energy_offsets,
    rx0,
    px0,
    hmax,
    hmin,
):
    curh = _np.full((energy_offsets.size, rx0.size), _np.inf)
    tune = _np.full((2, energy_offsets.size), _np.nan)
    ap_phys = _np.zeros(energy_offsets.size)
    ap_idx = _np.zeros(energy_offsets.size, dtype=int)
    beta = _np.ones(energy_offsets.size)
    twis = []
    for idx, delta in enumerate(energy_offsets):
        try:
            twi, *_ = _calc_twiss(
                accelerator, energy_offset=delta, indices='closed'
            )
            twis.append(twi)
            if _np.any(_np.isnan(twi[0].betax)):
                raise _OpticsError('error')
            tune[0, idx] = twi[-1].mux / (2*_np.pi)
            tune[1, idx] = twi[-1].muy / (2*_np.pi)
            beta[idx] = twi[0].betax
            rx = twi.rx
            px = twi.px
            betax = twi.betax
            dcox = rx - rx0
            dcoxp = px - px0
            curh[idx] = _get_curlyh(betax, twi.alphax, dcox, dcoxp)

            apper_loc = _np.minimum((hmax - rx)**2, (hmin - rx)**2)
            apper_loc /= betax
            ap_idx[idx] = _np.argmin(apper_loc)
            ap_phys[idx] = apper_loc[ap_idx[idx]]

        except (_OpticsError, _tracking.TrackingError):
            continue
    return curh, ap_phys, tune, beta, ap_idx, twis


def _calc_dyn_apert_for_touschek(
    accelerator,
    energies,
    curh,
    beta,
    nturns,
    parallel=False
):
    accelerator.cavity_on = False
    accelerator.radiation_on = 'off'
    accelerator.vchamber_on = False

    rin = _np.full((6, energies.size, curh.size), _np.nan)
    try:
        for idx, en in enumerate(energies):
            rin[:4, idx, :] = _tracking.find_orbit4(
                accelerator, energy_offset=en).ravel()[:, None]
    except _tracking.TrackingError:
        pass
    rin = rin.reshape(6, -1)

    accelerator.cavity_on = True
    accelerator.radiation_on = 'damping'
    accelerator.vchamber_on = True
    orb6d = _tracking.find_orbit6(accelerator)

    # Track positive energies
    curh0, ener = _np.meshgrid(curh, energies)
    xl = _np.sqrt(curh0/beta[:, None])

    rin[1, :] += xl.ravel()
    rin[2, :] += 1e-6
    rin[4, :] = orb6d[4] + ener.ravel()
    rin[5, :] = orb6d[5]

    _, loss_info = _tracking.ring_pass(
        accelerator, rin, nturns, turn_by_turn=False, parallel=parallel
    )
    lost = _np.reshape(loss_info.lost_flag, curh0.shape)
    ind_dyn = _np.argmax(lost, axis=1)
    ap_dyn = curh[ind_dyn]
    return ap_dyn
