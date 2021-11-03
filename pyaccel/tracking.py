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
import multiprocessing as _multiproc

import numpy as _np
import trackcpp as _trackcpp

import mathphys as _mp

from . import accelerator as _accelerator
from . import utils as _utils
from .utils import interactive as _interactive
from .optics.edwards_teng import EdwardsTeng


LOST_PLANES = (None, 'x', 'y', 'z')


class TrackingException(Exception):
    """."""


@_interactive
def generate_bunch(emitx, emity, sigmae, sigmas, twi, n_part, cutoff=3):
    """
    Create centered bunch with the desired equilibrium and twiss params.

    Inputs:
        emitx = horizontal emittance;
        emity = vertical emittance;
        sigmae = energy dispersion;
        sigmas = bunch length;
        twi = TwissObject at the desired location
        n_part = number of particles
        cutoff = number of sigmas to cut the distribution (in bunch size)

    Output:
        particles = numpy.array.shape == (6, n_part)
    """
    # avoiding notation confusion
    if type(twi) is EdwardsTeng:
        beta1 = twi.beta1
        beta2 = twi.beta2
        eta1 = twi.eta1
        eta2 = twi.eta2
        etap1 = twi.etap1
        etap2 = twi.etap2
        alpha1 = twi.alpha1
        alpha2 =  twi.alpha2
    else:
        beta1 = twi.betax
        beta2 = twi.betay
        eta1 = twi.etax
        eta2 = twi.etay
        etap1 = twi.etapx
        etap2 = twi.etapy
        alpha1 = twi.alphax
        alpha2 =  twi.alphay

    # generate longitudinal phase space
    parts = _mp.functions.generate_random_numbers(
        2*n_part, dist_type='norm', cutoff=cutoff)
    p_en = sigmae * parts[:n_part]
    p_s = sigmas * parts[n_part:]

    # generate transverse phase space
    parts = _mp.functions.generate_random_numbers(
        2*n_part, dist_type='exp', cutoff=cutoff*cutoff/2)
    amp1 = _np.sqrt(emitx * 2*parts[:n_part])
    amp2 = _np.sqrt(emity * 2*parts[n_part:])

    parts = _mp.functions.generate_random_numbers(
        2*n_part, dist_type='unif', cutoff=cutoff)
    ph1 = _np.pi * parts[:n_part]
    ph2 = _np.pi * parts[n_part:]

    p_1 = amp1*_np.sqrt(beta1)
    p_2 = amp2*_np.sqrt(beta2)
    p_1 *= _np.cos(ph1)
    p_2 *= _np.cos(ph2)
    p_1 += eta1 * p_en
    p_2 += eta2 * p_en
    p_1p = -amp1/_np.sqrt(beta1)
    p_2p = -amp2/_np.sqrt(beta2)
    p_1p *= alpha1*_np.cos(ph1) + _np.sin(ph1)
    p_2p *= alpha2*_np.cos(ph2) + _np.sin(ph2)
    p_1p += etap1 * p_en
    p_2p += etap2 * p_en
    return _np.array((p_1, p_1p, p_2, p_2p, p_en, p_s))


@_interactive
def set_4d_tracking(accelerator):
    accelerator.cavity_on = False
    accelerator.radiation_on = False


@_interactive
def set_6d_tracking(accelerator):
    accelerator.cavity_on = True
    accelerator.radiation_on = True


@_interactive
def element_pass(element, particles, energy, **kwargs):
    """Track particle(s) through an element.

    Accepts one or multiple particles initial positions. In the latter case,
    a list of particles or numpy 2D array (with particle as second index)
    should be given as input; also, outputs get an additional dimension,
    with particle as second index.

    Keyword arguments:
    element         -- 'Element' object
    particles       -- initial 6D particle(s) position(s)
                       ex.1: particles = [rx,px,ry,py,de,dl]
                       ex.3: particles = numpy.zeros((6, Np))
    energy          -- energy of the beam [eV]
    harmonic_number -- harmonic number of the lattice (optional, defaul=1)
    cavity_on       -- cavity on state (True/False) (optional, defaul=False)
    radiation_on    -- radiation on state (True/False) (optional, defaul=False)
    vchamber_on     -- vacuum chamber on state (True/False) (optional,
        defaul=False)

    Returns:
    part_out -- a numpy array with tracked 6D position(s) of the particle(s).
        If elementpass is invoked for a single particle then 'part_out' is a
        simple vector with one index that refers to the particle coordinates.
        If 'particles' represents many particles, the first index of
        'part_out' selects the coordinate and the second index selects the
        particle.

    Raises TrackingException
    """
    # checks if all necessary arguments have been passed
    kwargs['energy'] = energy

    # creates accelerator for tracking
    accelerator = _accelerator.Accelerator(**kwargs)

    # checks whether single or multiple particles
    p_in, _ = _process_args(accelerator, particles)

    # tracks through the list of pos
    ret = _trackcpp.track_elementpass_wrapper(
        element.trackcpp_e, p_in, accelerator.trackcpp_acc)
    if ret > 0:
        raise TrackingException

    return p_in.squeeze()


@_interactive
def line_pass(
        accelerator, particles, indices=None, element_offset=0,
        parallel=False):
    """Track particle(s) along a line.

    Accepts one or multiple particles initial positions. In the latter case,
    a list of particles or a numpy 2D array (with particle as second index)
    should be given as input; tracked particles positions at the entrances of
    elements are output variables, as well as information on whether particles
    have been lost along the tracking and where they were lost.

    Keyword arguments: (accelerator, particles, indices, element_offset)

    accelerator -- Accelerator object

    particles   -- initial 6D particle(s) position(s).
                   Few examples
                        ex.1: particles = [rx,px,ry,py,de,dl]
                        ex.2: particles = numpy.array([rx,px,ry,py,de,dl])
                        ex.3: particles = numpy.zeros((6, Np))

    indices     -- list of indices corresponding to accelerator elements at
                   whose entrances, tracked particles positions are to be
                   stored; string:
                   'open': corresponds to selecting all elements.
                   'closed' : equal 'open' plus the position at the end of the
                              last element.

    element_offset -- element offset (default 0) for tracking. tracking will
                      start at the element with index 'element_offset'

    parallel -- whether to parallelize calculation or not. If an integer is
                passed that many processes will be used. If True, the number
                of processes will be determined automatically.

    Returns: (part_out, lost_flag, lost_element, lost_plane)

    part_out -- 6D position for each particle at entrance of each element.
                The structure of 'part_out' depends on inputs
                'particles' and 'indices'. If 'indices' is None then only
                tracked positions at the end of the line are returned.
                There are still two possibilities for the structure of
                part_out, depending on 'particles':

                (1) if 'particles' is a single particle:
                    ex.:particles = [rx1,px1,ry1,py1,de1,dl1]
                        indices = None
                        part_out=numpy.array([rx2,px2,ry2,py2,de2,dl2])

                (2) if 'particles' is numpy matrix with several particles,
                    then 'part_out' will be a matrix (numpy array of
                    arrays) whose first index picks a coordinate rx, px,
                    ry, py, de or dl, in this order, and the second index
                    selects a particular particle.
                    ex.:particles.shape == (6, Np)
                        indices = None
                        part_out.shape == (6, Np)

                Now, if 'indices' is not None then 'part_out' can be either

                (3) a numpy matrix, when 'particles' is a single particle. The
                    first index of 'part_out' runs through the particle
                    coordinate and the second through the element index.

                (4) a numpy rank-3 tensor, when 'particles' is the initial
                    positions of many particles. The first index is the
                    coordinate index, the second index is the particle index
                    and the third index is the element index at whose
                    entrances particles coordinates are returned.

    lost_flag    -- a general flag indicating whether there has been particle
                    loss.
    lost_element -- list of element index where each particle was lost
                    If the particle survived the tracking through the line its
                    corresponding element in this list is set to None. When
                    there is only one particle defined as a python list (not as
                    a numpy matrix with one column) 'lost_element' returns a
                    single number.
    lost_plane   -- list of strings representing on what plane each particle
                    was lost while being tracked. If the particle is not lost
                    then its corresponding element in the list is set to None.
                    If it is lost in the horizontal or vertical plane it is set
                    to string 'x' or 'y', correspondingly. If tracking is
                    performed with a single particle described as a python list
                    then 'lost_plane' returns a single string

    """
    # checks whether single or multiple particles, reformats particles
    p_in, indices = _process_args(accelerator, particles, indices)
    indices = indices if indices is not None else [len(accelerator), ]

    if not parallel:
        p_out, lost_flag, lost_element, lost_plane = _line_pass(
            accelerator, p_in, indices, element_offset)
    else:
        slcs = _get_slices_multiprocessing(parallel, p_in.shape[1])
        with _multiproc.Pool(processes=len(slcs)) as pool:
            res = []
            for slc in slcs:
                res.append(pool.apply_async(_line_pass, (
                    accelerator, p_in[:, slc], indices, element_offset)))

            p_out, lost_element, lost_plane = [], [], []
            lost_flag = False
            for re_ in res:
                part_out, lflag, lelement, lplane = re_.get()
                lost_flag |= lflag
                p_out.append(part_out)
                lost_element.extend(lelement)
                lost_plane.extend(lplane)
        p_out = _np.concatenate(p_out, axis=1)

    # simplifies output structure in case of single particle
    if len(lost_element) == 1:
        lost_element = lost_element[0]
        lost_plane = lost_plane[0]

    return p_out, lost_flag, lost_element, lost_plane


def _line_pass(accelerator, p_in, indices, element_offset):
    # store only final position?
    args = _trackcpp.LinePassArgs()
    for idx in indices:
        args.indices.push_back(int(idx))
    args.element_offset = int(element_offset)

    n_part = p_in.shape[1]
    p_out = _np.zeros((6, n_part * len(indices)), dtype=float)

    # tracking
    lost_flag = bool(_trackcpp.track_linepass_wrapper(
        accelerator.trackcpp_acc, p_in, p_out, args))

    p_out = p_out.reshape(6, n_part, -1)
    p_out = _np.squeeze(p_out)

    # fills vectors with info about particle loss
    lost_element = list(args.lost_element)
    lost_plane = [LOST_PLANES[lp] for lp in args.lost_plane]

    return p_out, lost_flag, lost_element, lost_plane


@_interactive
def ring_pass(
        accelerator, particles, nr_turns=1, turn_by_turn=None,
        element_offset=0, parallel=False):
    """Track particle(s) along a ring.

    Accepts one or multiple particles initial positions. In the latter case,
    a list of particles or a numpy 2D array (with particle as firts index)
    should be given as input; tracked particles positions at the end of
    the ring are output variables, as well as information on whether particles
    have been lost along the tracking and where they were lost.

    Keyword arguments: (accelerator, particles, nr_turns,
                        turn_by_turn, elment_offset)

    accelerator    -- Accelerator object
    particles      -- initial 6D particle(s) position(s).
                      Few examples
                        ex.1: particles = [rx,px,ry,py,de,dl]
                        ex.2: particles = numpy.array([rx,px,ry,py,de,dl])
                        ex.3: particles = numpy.zeros((6, Np))
    nr_turns       -- number of turns around ring to track each particle.
    turn_by_turn   -- parameter indicating what turn by turn positions are to
                      be returned. If None ringpass returns particles
                      positions only at the end of the ring, at the last turn.
                      If bool(turn_by_turn) is True, ringpass returns positions
                      at the beginning of every turn (including the first) and
                      at the end of the ring in the last turn.

    element_offset -- element offset (default 0) for tracking. tracking will
                      start at the element with index 'element_offset'

    parallel -- whether to parallelize calculation or not. If an integer is
                passed that many processes will be used. If True, the number
                of processes will be determined automatically.

    Returns: (part_out, lost_flag, lost_turn, lost_element, lost_plane)

    part_out -- 6D position for each particle at end of ring. The
                     structure of 'part_out' depends on inputs
                     'particles' and 'turn_by_turn'. If 'turn_by_turn' is None
                     then only tracked positions at the end 'nr_turns' are
                     returned. There are still two possibilities for the
                     structure of part_out, depending on 'particles':

                    (1) if 'particles' is a single particle, 'part_out' will
                        also be a unidimensional numpy array:
                        ex.:particles = [rx1,px1,ry1,py1,de1,dl1]
                            turn_by_turn = False
                            part_out = numpy.array([rx2,px2,ry2,py2,de2,dl2])

                    (2) if 'particles' is either a numpy matrix with several
                        particles then 'part_out' will be a matrix (numpy
                        array of arrays) whose first index selects the
                        coordinate rx, px, ry, py, de or dl, in this order,
                        and the second index selects a particular particle.
                        ex.: particles.shape == (6, Np)
                             turn_by_turn = False
                             part_out.shape == (6, Np))

                     'turn_by_turn' can also be either 'closed' or 'open'. In
                     either case 'part_out' will have tracked positions at
                     the entrances of the elements. The difference is that for
                     'closed' it will have an additional tracked position at
                     the exit of the last element, thus closing the data, in
                     case the line is a ring. The format of 'part_out' is
                     ...

                    (3) a numpy matrix, when 'particles' is a single particle.
                        The first index of 'part_out' runs through coordinates
                        rx, px, ry, py, de or dl and the second index runs
                        through the turn number.

                    (4) a numpy rank-3 tensor, when 'particles' is the initial
                        positions of many particles. The first index runs
                        through coordinates, the second through particles and
                        the third through turn number.

    lost_flag    -- a general flag indicating whether there has been particle
                    loss.
    lost_turn    -- list of turn index where each particle was lost.
    lost_element -- list of element index where each particle was lost
                    If the particle survived the tracking through the ring its
                    corresponding element in this list is set to None. When
                    there is only one particle defined as a python list (not as
                    a numpy matrix with one column) 'lost_element' returns a
                    single number.
    lost_plane   -- list of strings representing on what plane each particle
                    was lost while being tracked. If the particle is not lost
                    then its corresponding element in the list is set to None.
                    If it is lost in the horizontal or vertical plane it is set
                    to string 'x' or 'y', correspondingly. If tracking is
                    performed with a single particle described as a python list
                    then 'lost_plane' returns a single string.
    """
    # checks whether single or multiple particles, reformats particles
    p_in, *_ = _process_args(accelerator, particles, indices=None)

    if not parallel:
        p_out, lost_flag, lost_turn, lost_element, lost_plane = _ring_pass(
            accelerator, p_in, nr_turns, turn_by_turn, element_offset)
    else:
        slcs = _get_slices_multiprocessing(parallel, p_in.shape[1])
        with _multiproc.Pool(processes=len(slcs)) as pool:
            res = []
            for slc in slcs:
                res.append(pool.apply_async(_ring_pass, (
                    accelerator, p_in[:, slc], nr_turns, turn_by_turn,
                    element_offset)))

            p_out, lost_turn, lost_element, lost_plane = [], [], [], []
            lost_flag = False
            for re_ in res:
                part_out, lflag, lturn, lelement, lplane = re_.get()
                lost_flag |= lflag
                p_out.append(part_out)
                lost_turn.extend(lturn)
                lost_element.extend(lelement)
                lost_plane.extend(lplane)
        p_out = _np.concatenate(p_out, axis=1)

    # simplifies output structure in case of single particle
    if len(lost_element) == 1:
        lost_turn = lost_turn[0]
        lost_element = lost_element[0]
        lost_plane = lost_plane[0]

    return p_out, lost_flag, lost_turn, lost_element, lost_plane


def _ring_pass(accelerator, p_in, nr_turns, turn_by_turn, element_offset):
    # static parameters of ringpass
    args = _trackcpp.RingPassArgs()
    args.nr_turns = int(nr_turns)
    args.trajectory = bool(turn_by_turn)
    args.element_offset = int(element_offset)

    n_part = p_in.shape[1]
    if bool(turn_by_turn):
        p_out = _np.zeros((6, n_part*(nr_turns+1)), dtype=float)
    else:
        p_out = _np.zeros((6, n_part), dtype=float)

    # tracking
    lost_flag = bool(_trackcpp.track_ringpass_wrapper(
        accelerator.trackcpp_acc, p_in, p_out, args))

    p_out = p_out.reshape(6, n_part, -1)
    p_out = _np.squeeze(p_out)

    # fills vectors with info about particle loss
    lost_turn = list(args.lost_turn)
    lost_element = list(args.lost_element)
    lost_plane = [LOST_PLANES[lp] for lp in args.lost_plane]

    return p_out, lost_flag, lost_turn, lost_element, lost_plane


@_interactive
def find_orbit4(accelerator, energy_offset=0.0, indices=None,
                fixed_point_guess=None):
    """Calculate 4D closed orbit of accelerator and return it.

    Accepts an optional list of indices of ring elements where closed orbit
    coordinates are to be returned. If this argument is not passed, closed
    orbit positions are returned at the start of the first element. In
    addition a guess fixed point at the entrance of the ring may be provided.

    Keyword arguments:
    accelerator : Accelerator object
    energy_offset : relative energy deviation from nominal energy
    indices : may be a (list,tuple, numpy.ndarray) of element indices where
        closed orbit data is to be returned or a string:
            'open'  : return the closed orbit at the entrance of all elements.
            'closed' : equal 'open' plus the orbit at the end of the last
                element.
        If indices is None the closed orbit is returned only at the entrance
        of the first element.
    fixed_point_guess : A 6D position where to start the search of the closed
        orbit at the entrance of the first element. If not provided the
        algorithm will start with zero orbit.

    Returns:
    orbit : 4D closed orbit at the entrance of the selected elements as a 2D
        numpy array with the 4 phase space variables in the first dimension and
        the indices of the elements in the second dimension.

    Raises TrackingException
    """
    indices = _process_indices(accelerator, indices)

    if fixed_point_guess is None:
        fixed_point_guess = _trackcpp.CppDoublePos()
    else:
        fixed_point_guess = _4Numpy2CppDoublePos(fixed_point_guess)
    fixed_point_guess.de = energy_offset

    _closed_orbit = _trackcpp.CppDoublePosVector()
    ret = _trackcpp.track_findorbit4(
        accelerator.trackcpp_acc, _closed_orbit, fixed_point_guess)
    if ret > 0:
        raise TrackingException(_trackcpp.string_error_messages[ret])

    closed_orbit = _CppDoublePosVector24Numpy(_closed_orbit)
    return closed_orbit[:, indices]


@_interactive
def find_orbit6(accelerator, indices=None, fixed_point_guess=None):
    """Calculate 6D closed orbit of accelerator and return it.

    Accepts an optional list of indices of ring elements where closed orbit
    coordinates are to be returned. If this argument is not passed, closed
    orbit positions are returned at the start of the first element. In
    addition a guess fixed point at the entrance of the ring may be provided.

    Keyword arguments:
    accelerator : Accelerator object
    indices : may be a (list,tuple, numpy.ndarray) of element indices
        where closed orbit data is to be returned or a string:
            'open'  : return the closed orbit at the entrance of all elements.
            'closed' : equal 'open' plus the orbit at the end of the last
                element.
        If indices is None the closed orbit is returned only at the entrance
        of the first element.
    fixed_point_guess : A 6D position where to start the search of the closed
        orbit at the entrance of the first element. If not provided the
        algorithm will start with zero orbit.

    Returns:
        orbit : 6D closed orbit at the entrance of the selected elements as
            a 2D numpy array with the 6 phase space variables in the first
            dimension and the indices of the elements in the second dimension.

    Raises TrackingException

    """
    indices = _process_indices(accelerator, indices)

    if fixed_point_guess is None:
        fixed_point_guess = _trackcpp.CppDoublePos()
    else:
        fixed_point_guess = _Numpy2CppDoublePos(fixed_point_guess)

    _closed_orbit = _trackcpp.CppDoublePosVector()

    ret = _trackcpp.track_findorbit6(
        accelerator.trackcpp_acc, _closed_orbit, fixed_point_guess)
    if ret > 0:
        raise TrackingException(_trackcpp.string_error_messages[ret])

    closed_orbit = _CppDoublePosVector2Numpy(_closed_orbit)
    return closed_orbit[:, indices]


@_interactive
def find_orbit(
        accelerator, energy_offset=None, indices=None, fixed_point_guess=None):
    """Calculate 6D closed orbit of accelerator and return it.

    Automatically identifies if find_orbit4 or find_orbit6 must be used based
    on the state of `radiation_on` and `cavity_on` properties of accelerator.

    Accepts an optional list of indices of ring elements where closed orbit
    coordinates are to be returned. If this argument is not passed, closed
    orbit positions are returned at the start of the first element. In
    addition a guess fixed point at the entrance of the ring may be provided.

    Keyword arguments:
    accelerator : Accelerator object
    indices : may be a (list,tuple, numpy.ndarray) of element indices
        where closed orbit data is to be returned or a string:
            'open'  : return the closed orbit at the entrance of all elements.
            'closed' : equal 'open' plus the orbit at the end of the last
                element.
        If indices is None the closed orbit is returned only at the entrance
        of the first element.
    fixed_point_guess : A 6D position where to start the search of the closed
        orbit at the entrance of the first element. If not provided the
        algorithm will start with zero orbit.

    Returns:
        orbit : 6D closed orbit at the entrance of the selected elements as
            a 2D numpy array with the 6 phase space variables in the first
            dimension and the indices of the elements in the second dimension.

    Raises TrackingException

    """
    if not accelerator.cavity_on and not accelerator.radiation_on:
        energy_offset = energy_offset or 0.0
        orb = find_orbit4(
            accelerator, indices=indices, energy_offset=energy_offset,
            fixed_point_guess=fixed_point_guess)
        corb = _np.zeros((6, orb.shape[1]))
        corb[:4, :] = orb
        corb[4, :] = energy_offset
        return corb
    elif not accelerator.cavity_on and accelerator.radiation_on:
        raise TrackingException('The radiation is on but the cavity is off')
    else:
        return find_orbit6(
            accelerator, indices=indices, fixed_point_guess=fixed_point_guess)


@_interactive
def find_m66(accelerator, indices='m66', fixed_point=None):
    """Calculate 6D transfer matrices of elements in an accelerator.

    Keyword arguments:
    accelerator : Accelerator object
    indices : may be a (list, tuple, numpy.ndarray) of element indices where
              cumul_trans_matrices is to be returned or a string:
               'open' : return the cumul_trans_matrices at the entrance of all
                         elements.
               'closed' : equal 'open' plus the matrix at the end of the last
                        element.
               'm66'  : the cumul_trans_matrices is not returned.
              If indices is None the cumul_trans_matrices is not returned.
    fixed_point (numpy.ndarray, (6, )): phase space position at the start of
        the lattice where the matrices will be calculated around.

    Return values:
    m66
    cumul_trans_matrices -- values at the start of each lattice element

    """
    if isinstance(indices, str) and indices == 'm66':
        indices = None
    indices = _process_indices(accelerator, indices, proc_none=False)

    trackcpp_idx = _trackcpp.CppUnsigIntVector()
    if isinstance(indices, _np.ndarray):
        trackcpp_idx.reserve(indices.size)
        for i in indices:
            trackcpp_idx.push_back(int(i))
    else:
        trackcpp_idx.push_back(len(accelerator))

    if fixed_point is None:
        # Closed orbit is calculated by trackcpp
        fixed_point_guess = _trackcpp.CppDoublePos()
        _closed_orbit = _trackcpp.CppDoublePosVector()
        ret = _trackcpp.track_findorbit6(
            accelerator.trackcpp_acc, _closed_orbit, fixed_point_guess)
        if ret > 0:
            raise TrackingException(_trackcpp.string_error_messages[ret])
    else:
        _fixed_point = _Numpy2CppDoublePos(fixed_point)
        _closed_orbit = _trackcpp.CppDoublePosVector()
        _closed_orbit.push_back(_fixed_point)

    cumul_trans_matrices = _np.zeros((trackcpp_idx.size(), 6, 6), dtype=float)
    m66 = _np.zeros((6, 6), dtype=float)
    _v0 = _trackcpp.CppDoublePos()
    ret = _trackcpp.track_findm66_wrapper(
        accelerator.trackcpp_acc, _closed_orbit[0], cumul_trans_matrices,
        m66, _v0, trackcpp_idx)
    if ret > 0:
        raise TrackingException(_trackcpp.string_error_messages[ret])

    if indices is None:
        return m66
    return m66, cumul_trans_matrices


@_interactive
def find_m44(accelerator, indices='m44', energy_offset=0.0, fixed_point=None):
    """Calculate 4D transfer matrices of elements in an accelerator.

    Keyword arguments:
    accelerator : Accelerator object
    indices : may be a (list, tuple, numpy.ndarray) of element indices where
              cumul_trans_matrices is to be returned or a string:
               'open' : return the cumul_trans_matrices at the entrance of all
                         elements.
                'closed' : equal 'open' plus the matrix at the end of the last
                        element.
               'm44'  : the cumul_trans_matrices is not returned.
              If indices is None the cumul_trans_matrices is not returned.
    energy_offset (float, ): energy offset
    fixed_point (numpy.ndarray, (4, )): phase space position at the start of
        the lattice where the matrices will be calculated around.

    Return values:
    m44
    cumul_trans_matrices -- values at the start of each lattice element
    """
    if isinstance(indices, str) and indices == 'm44':
        indices = None
    indices = _process_indices(accelerator, indices, proc_none=False)

    trackcpp_idx = _trackcpp.CppUnsigIntVector()
    if isinstance(indices, _np.ndarray):
        trackcpp_idx.reserve(indices.size)
        for i in indices:
            trackcpp_idx.push_back(int(i))
    else:
        trackcpp_idx.push_back(len(accelerator))

    if fixed_point is None:
        # calcs closed orbit if it was not passed.
        fixed_point_guess = _trackcpp.CppDoublePos()
        fixed_point_guess.de = energy_offset
        _closed_orbit = _trackcpp.CppDoublePosVector()
        ret = _trackcpp.track_findorbit4(
            accelerator.trackcpp_acc, _closed_orbit, fixed_point_guess)

        if ret > 0:
            raise TrackingException(_trackcpp.string_error_messages[ret])
    else:
        _fixed_point = _4Numpy2CppDoublePos(fixed_point, de=energy_offset)
        _closed_orbit = _trackcpp.CppDoublePosVector()
        _closed_orbit.push_back(_fixed_point)

    cumul_trans_matrices = _np.zeros((trackcpp_idx.size(), 4, 4), dtype=float)
    m44 = _np.zeros((4, 4), dtype=float)
    _v0 = _trackcpp.CppDoublePos()
    ret = _trackcpp.track_findm66_wrapper(
        accelerator.trackcpp_acc, _closed_orbit[0], cumul_trans_matrices,
        m44, _v0, trackcpp_idx)
    if ret > 0:
        raise TrackingException(_trackcpp.string_error_messages[ret])

    if indices is None:
        return m44
    return m44, cumul_trans_matrices


# ------ Auxiliary methods -------

def _get_slices_multiprocessing(parallel, nparticles):
    nrproc = _multiproc.cpu_count() - 3
    nrproc = nrproc if parallel is True else parallel
    nrproc = max(nrproc, 1)
    nrproc = min(nrproc, nparticles)

    np_proc = (nparticles // nrproc)*_np.ones(nrproc, dtype=int)
    np_proc[:(nparticles % nrproc)] += 1
    parts_proc = _np.r_[0, _np.cumsum(np_proc)]
    return [slice(parts_proc[i], parts_proc[i+1]) for i in range(nrproc)]


def _CppMatrix2Numpy(_m):
    return _np.array(_m)


def _CppMatrix24Numpy(_m):
    return _np.array(_m)[:4, :4]


def _Numpy2CppDoublePos(p_in):
    p_out = _trackcpp.CppDoublePos(
        float(p_in[0]), float(p_in[1]),
        float(p_in[2]), float(p_in[3]),
        float(p_in[4]), float(p_in[5]))
    return p_out


def _4Numpy2CppDoublePos(p_in, de=0.0):
    p_out = _trackcpp.CppDoublePos(
        float(p_in[0]), float(p_in[1]),
        float(p_in[2]), float(p_in[3]),
        de, 0)
    return p_out


def _CppDoublePos2Numpy(p_in):
    return _np.array((p_in.rx, p_in.px, p_in.ry, p_in.py, p_in.de, p_in.dl))


def _CppDoublePos24Numpy(p_in):
    return _np.array((p_in.rx, p_in.px, p_in.ry, p_in.py))


def _Numpy2CppDoublePosVector(poss):
    if isinstance(poss, _trackcpp.CppDoublePosVector):
        return poss

    conds = isinstance(poss, _np.ndarray)
    conds &= len(poss.shape) == 2 and poss.shape[0] == 6
    if not conds:
        raise TrackingException('invalid positions argument')

    poss_out = _trackcpp.CppDoublePosVector()
    for pos in poss.T:
        poss_out.push_back(_Numpy2CppDoublePos(pos))
    return poss_out


def _4Numpy2CppDoublePosVector(poss, de=0.0):
    if isinstance(poss, _trackcpp.CppDoublePosVector):
        return poss

    conds = isinstance(poss, _np.ndarray)
    conds &= len(poss.shape) == 2 and poss.shape[0] == 4
    if not conds:
        raise TrackingException('invalid positions argument')

    poss_out = _trackcpp.CppDoublePosVector()
    for pos in poss.T:
        poss_out.push_back(_4Numpy2CppDoublePos(pos, de=de))
    return poss_out


def _CppDoublePosVector2Numpy(poss):
    if not isinstance(poss, _trackcpp.CppDoublePosVector):
        raise TrackingException('invalid positions argument')

    poss_out = _np.zeros((6, poss.size()))
    for i, pos in enumerate(poss):
        poss_out[:, i] = _CppDoublePos2Numpy(pos)
    return poss_out


def _CppDoublePosVector24Numpy(poss):
    if not isinstance(poss, _trackcpp.CppDoublePosVector):
        raise TrackingException('invalid positions argument')

    poss_out = _np.zeros((4, poss.size()))
    for i, pos in enumerate(poss):
        poss_out[:, i] = _CppDoublePos24Numpy(pos)
    return poss_out


def _process_args(accelerator, pos, indices=None, dim='6d'):
    pos = _process_array(pos, dim=dim)
    indices = _process_indices(accelerator, indices, proc_none=False)
    return pos, indices


def _process_array(pos, dim='6d'):
    # checks whether single or multiple particles
    if isinstance(pos, (list, tuple)):
        if isinstance(pos[0], (list, tuple)):
            pos = _np.array(pos).T
        else:
            pos = _np.array(pos, ndmin=2).T
    elif isinstance(pos, _np.ndarray):
        if dim not in ('4d', '6d'):
            raise TrackingException('dimension argument must be 4d or 6d.')
        posdim = 4 if dim == '4d' else 6
        if len(pos.shape) == 1:
            pos = _np.array(pos, ndmin=2).T
        elif len(pos.shape) > 2 or pos.shape[0] != posdim:
            raise TrackingException('invalid position argument.')
    return pos


def _process_indices(accelerator, indices, closed=True, proc_none=True):
    if indices is None:
        indices = [0, ] if proc_none else indices
    elif isinstance(indices, str):
        if indices.startswith('open'):
            indices = _np.arange(len(accelerator))
        elif closed and indices.startswith('closed'):
            indices = _np.arange(len(accelerator)+1)
        else:
            raise TrackingException("invalid value for 'indices'")
    elif isinstance(indices, (list, tuple, _np.ndarray)):
        try:
            indices = _np.array(indices, dtype=int)
        except ValueError:
            raise TrackingException("invalid value for 'indices'")
        if len(indices.shape) > 1:
            raise TrackingException("invalid value for 'indices'")
    else:
        raise TrackingException("invalid value for 'indices'")
    return indices


def _print_CppDoublePos(pos):
    print('\n{0:+.16f}'.format(pos.rx))
    print('{0:+.16f}'.format(pos.px))
    print('{0:+.16f}'.format(pos.ry))
    print('{0:+.16f}'.format(pos.py))
    print('{0:+.16f}'.format(pos.de))
    print('{0:+.16f}\n'.format(pos.dl))


# Legacy API
elementpass = _utils.deprecated(element_pass)
linepass = _utils.deprecated(line_pass)
ringpass = _utils.deprecated(ring_pass)
set4dtracking = _utils.deprecated(set_4d_tracking)
set6dtracking = _utils.deprecated(set_6d_tracking)
findorbit4 = _utils.deprecated(find_orbit4)
findorbit6 = _utils.deprecated(find_orbit6)
findm66 = _utils.deprecated(find_m66)
findm44 = _utils.deprecated(find_m44)
