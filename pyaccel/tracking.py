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

import mathphys as _mp
import numpy as _np
import trackcpp as _trackcpp

from .accelerator import Accelerator as _Accelerator
from .optics.edwards_teng import EdwardsTeng as _EdwardsTeng
from .optics.twiss import Twiss as _Twiss
from .utils import interactive as _interactive


class TrackingError(Exception):
    """."""


class TrackingLossInfo:
    """Class with useful information about lost particles."""

    # See trackcpp.Plane
    LostPlanes = _np.array(['no_plane', 'x', 'y', 'z', 'xy'], dtype=str)

    def __init__(
        self,
        lost_flag: list,
        lost_plane: list,
        lost_element: list,
        lost_pos: _np.ndarray,
        lost_turn: list = None,
    ):
        """Information about particles lost in tracking.

        Args:
            lost_flag (list): list of booleans indicating whether each particle
                was lost.
            lost_element (list): list of element index where each particle was
                lost. If the particle survived the tracking through the line
                its corresponding element in this list is set to -1.
            lost_plane (list): list of integers representing on what plane each
                particle was lost while being tracked.
            lost_pos (numpy.ndarray, (6, Np)): 6D position vector of each lost
                particle at the moment they were lost. Position is set to NaN
                when particle is not lost.
            lost_turn (list, optional): list of turn index where each particle
                was lost. Is set to -1 if particle was not lost. Defaults to
                None. If None, then associated attribute will not be created.

        """
        self.lost_flag = _np.ndarray(lost_flag)
        self.lost_plane = _np.ndarray(lost_plane)
        self.lost_element = _np.ndarray(lost_element)
        self.lost_pos = lost_pos
        if lost_turn is not None:
            self.lost_turn = lost_turn

    @property
    def lost_plane_str(self):
        """Return numpy array of strings indicating the lost plane."""
        return self.LostPlanes[self.lost_plane]

    @property
    def num_particles(self):
        """Returns number of tracked particles.

        Returns:
            num_particles (int): number of tracked particles.
        """
        return len(self.lost_flag)

    @property
    def num_lost_particles(self):
        """Returns number of lost particles.

        Returns:
            num_lost_particles (int): number of lost particles.
        """
        return self.num_particles - sum(self.lost_flag)

    @property
    def indices_lost_particles(self):
        """Returns indices of lost particles.

        Returns:
            indices_lost_particles (numpy.ndarray): indices of lost particles.
        """
        return _np.array(self.lost_flag).nonzero()[0]


@_interactive
def generate_bunch(
        n_part, envelope=None, emit1=None, emit2=None, sigmae=None,
        sigmas=None, optics=None, cutoff=3):
    """Create centered bunch with the desired equilibrium and optics params.

    Args:
        n_part (int): number of particles;
        envelope (numpy.array, optional): 6x6 matrix with beam second moments.
            when provided, preceeds the usage of the equilibrium parameters.
            Otherwise they are mandatory. Defaults to None.
        emit1 (float, optional): first mode emittance, corresponds to
            horizontal emittance at the limit of zero coupling.
            Not used when envelope is provided, mandatory otherwise.
            Defaults to None.
        emit2 (float, optional): second mode emittance, corresponds to
            vertical emittance at the limit of zero coupling.
            Not used when envelope is provided, mandatory otherwise.
            Defaults to None.
        sigmae (float, optional): energy dispersion. Not used when envelope is
            provided, mandatory otherwise. Defaults to None.
        sigmas (float, optional): bunch length. Not used when envelope is
            provided, mandatory otherwise. Defaults to None.
        optics (pyaccel.optics.Twiss or pyaccel.optics.EdwardsTeng, optional):
            Optical functions at desired location. Not used when envelope is
            provided, mandatory otherwise. Defaults to None.
        cutoff (int, optional): number of sigmas to cut the distribution.
            Defaults to 3.

    Raises:
        TypeError: raised if optics is not a pyaccel.optics.Twiss or
            pyaccel.optics.EdwardsTeng object.
        ValueError: raised when envelope is not provided and not all
            equilibrium parameters are defined.

    Returns:
        numpy.ndarray: 6 x n_part array where each column is a particle.

    """
    if envelope is not None:
        # The method used below was based on the algorithm described here:
        #    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        # in section: "4.1 Drawing_values_from_the_distribution"
        #
        # Create 6D vectors whose components follow the normal
        # distribution, such that:
        # np.cov(znor) == <znor @ znor.T> == np.eye(6)
        znor = _mp.functions.generate_random_numbers(
            6*n_part, dist_type='norm', cutoff=cutoff).reshape(6, -1)
        # The Cholesky decomposition finds a matrix env_chol such that:
        # env_chol @ env_chol.T == envelope
        try:
            env_chol = _np.linalg.cholesky(envelope)
        except _np.linalg.LinAlgError:
            # In case there is no coupling the envelope matrix is only
            # positive semi-definite and we must disconsider the vertical
            # plane for the decomposition algorithm to work:
            env_chol = _np.zeros((6, 6))
            idx = _np.ix_([0, 1, 4, 5], [0, 1, 4, 5])
            env_chol[idx] = _np.linalg.cholesky(envelope[idx])
        # This way we can find bunch through:
        bunch = env_chol @ znor
        # where one can see that:
        # np.cov(bunch) == <bunch @ bunch.T> ==
        #     env_chol @ <znor @ znor.T> @ env_chol.T ==
        #     env_chol @ np.eye(6) @ env_chol.T ==
        #     env_chol @ env_chol.T == envelope
        return bunch

    if None in (emit1, emit2, sigmae, sigmas, optics):
        raise ValueError(
            'When envelope is not provided, emit1, emit2, sigmae, sigmas ' +
            'and optics are mandatory arguments.')

    if isinstance(optics, _Twiss):
        beta1, beta2 = optics.betax, optics.betay
        alpha1, alpha2 = optics.alphax, optics.alphay
        eta1, eta2 = optics.etax, optics.etay
        etap1, etap2 = optics.etapx, optics.etapy
    elif isinstance(optics, _EdwardsTeng):
        beta1, beta2 = optics.beta1, optics.beta2
        alpha1, alpha2 = optics.alpha1, optics.alpha2
        eta1, eta2 = optics.eta1, optics.eta2
        etap1, etap2 = optics.etap1, optics.etap2
    else:
        raise TypeError('optics arg must be a Twiss or EdwardsTeng object.')

    # generate longitudinal phase space
    parts = _mp.functions.generate_random_numbers(
        2*n_part, dist_type='norm', cutoff=cutoff)
    p_en = sigmae * parts[:n_part]
    p_s = sigmas * parts[n_part:]

    # generate transverse phase space
    parts = _mp.functions.generate_random_numbers(
        2*n_part, dist_type='exp', cutoff=cutoff*cutoff/2)
    amp1 = _np.sqrt(emit1 * 2*parts[:n_part])
    amp2 = _np.sqrt(emit2 * 2*parts[n_part:])

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

    # bunch at normal modes coordinates:
    bunch_nm = _np.array((p_1, p_1p, p_2, p_2p, p_en, p_s))

    if isinstance(optics, _EdwardsTeng):
        # bunch at (x, x', y, y', de, dl) coordinates:
        bunch = optics.from_normal_modes(bunch_nm)
    else:
        bunch = bunch_nm

    return bunch


@_interactive
def set_4d_tracking(accelerator: _Accelerator):
    """Turn off radiation and cavity of accelerator.

    Args:
        accelerator (pyaccel.accelerator.Accelerator): accelerator object.
    """
    accelerator.cavity_on = False
    accelerator.radiation_on = 'Off'


@_interactive
def set_6d_tracking(accelerator: _Accelerator, rad_full=False):
    """Turn on radiation and cavity of accelerator.

    Args:
        accelerator (_type_): accelerator object.
        rad_full (bool, optional): whether to consider quantum excitation in
            tracking. Defaults to False.
    """
    accelerator.cavity_on = True
    accelerator.radiation_on = 'Full' if rad_full else 'Damping'


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
    harmonic_number -- harmonic number of the lattice (optional, default=1)
    cavity_on       -- cavity on state (True/False) (optional, default=False)
    radiation_on    -- radiation on state (0 or "off", 1 or "damping", 2 or
        "full") (optional, default="off")
    vchamber_on     -- vacuum chamber on state (True/False) (optional,
        default=False)

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
        accelerator.trackcpp_acc, element.trackcpp_e, p_in
    )
    if ret > 0:
        raise TrackingError("Problem found during tracking.")

    return p_in.squeeze()


@_interactive
def line_pass(
    accelerator,
    particles,
    indices=None,
    element_offset=0,
    parallel=False
):
    """Track particle(s) along a line.

    Accepts one or multiple particles initial positions. In the latter case,
    a list of particles or a numpy 2D array (with particle as second index)
    should be given as input; tracked particles positions at the entrances of
    elements are output variables, as well as information on whether particles
    have been lost along the tracking and where they were lost.

    Args:
        accelerator (pyaccel.accelerator.Accelerator): accelerator object.
        particles (list|numpy.ndarray): initial 6D particle(s) position(s):
            Few examples
                ex.1: particles = [rx,px,ry,py,de,dl]
                ex.2: particles = numpy.array([rx,px,ry,py,de,dl])
                ex.3: particles = numpy.zeros((6, Np))
        indices (list|str|tuple|numpy.ndarray, optional): list of indices
            corresponding to accelerator elements at whose entrances, tracked
            particles positions are to be stored. If string:
                'open': corresponds to selecting all elements.
                'closed' : equal 'open' plus the position at the end of the
                    last element.
            Defaults to None, which means only the position at the end of the
            last element will be returned.
        element_offset (int, optional): element offset (default 0) for
            tracking. Tracking will start at the element with index
            'element_offset'. Defaults to 0.
        parallel (bool, optional): whether to parallelize calculation or not.
            If an integer is passed that many processes will be used. If True,
            the number of processes will be determined automatically.
            Defaults to False.

    Returns:
        part_out (numpy.ndarray): 6D position for each particle at entrance of
            each element. The structure of 'part_out' depends on inputs
            'particles' and 'indices'. If 'indices' is None then only tracked
            positions at the end of the line are returned. There are still two
            possibilities for the structure of part_out, depending on
            'particles':
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
        loss_info (pyaccel.tracking.TrackingLossInfo): object with useful
            information about lost particles. Contains the following
            attributes:
            lost_flag (numpy.ndarray, dtype=bool): list of booleans indicating
                whether each particle was lost;
            lost_element (numpy.ndarray, dtype=int): list of element index
                where each particle was lost. If the particle survived the
                tracking through the line its corresponding element in this
                list is set to -1;
            lost_plane (numpy.ndarray, dtype=int): list of strings
                representing on what plane each particle was lost while being
                tracked. If the particle is not lost then its corresponding
                element in the list is set to None;
            lost_plane_str (numpy.ndarray, dtype=str): Same as previous but
                returns the string interpretation of the lost plane of each
                particles. Possible values are
                    ('no_plane', 'x', 'y', 'z', 'xy');
            lost_pos (numpy.ndarray, (6, Np)): 6D position vector of each lost
                particle at the moment they were lost. Position is set to NaN
                when particle is not lost;
            nr_particles (int): number of tracked particles;
            nr_lost_particles (int): number of lost particles;
            indices_lost_particles (numpy.ndarray, dtype=int): indices of lost
                particles.

    """
    # checks whether single or multiple particles, reformats particles
    p_in, indices = _process_args(accelerator, particles, indices)
    indices = indices if indices is not None else [len(accelerator), ]

    if not parallel:
        p_out, lost_flag, lost_element, lost_plane, lost_pos = _line_pass(
            accelerator, p_in, indices, element_offset)
    else:
        slcs = _get_slices_multiprocessing(parallel, p_in.shape[1])
        with _multiproc.Pool(processes=len(slcs)) as pool:
            res = []
            for slc in slcs:
                res.append(pool.apply_async(_line_pass, (
                    accelerator, p_in[:, slc], indices, element_offset, True)))

            lost_pos, p_out = [], []
            lost_flag, lost_element, lost_plane = [], [], []
            for re_ in res:
                part_out, lflag, lelement, lplane, part_lost = re_.get()
                p_out.append(part_out)
                lost_pos.append(part_lost)
                lost_flag.extend(lflag)
                lost_element.extend(lelement)
                lost_plane.extend(lplane)
        p_out = _np.concatenate(p_out, axis=1)
        lost_pos = _np.concatenate(lost_pos, axis=1)

    # simplifies output structure in case of single particle
    p_out = _np.squeeze(p_out)
    lost_pos = _np.squeeze(lost_pos)
    loss_info = TrackingLossInfo(lost_flag, lost_plane, lost_element, lost_pos)
    return p_out, loss_info


def _line_pass(accelerator, p_in, indices, element_offset, set_seed=False):
    # store only final position?
    args = _trackcpp.LinePassArgs()
    for idx in indices:
        args.indices.push_back(int(idx))
    args.element_offset = int(element_offset)

    p_in = p_in.copy()
    n_part = p_in.shape[1]
    p_out = _np.zeros((6, n_part * len(indices)), dtype=float)

    # re-seed pseudo-random generator
    if set_seed:
        _set_random_seed()

    # tracking
    lost_flag = bool(_trackcpp.track_linepass_wrapper(
        accelerator.trackcpp_acc, p_in, p_out, args))

    # After tracking, the input vector contains the lost positions.
    lost_pos = p_in
    p_out = p_out.reshape(6, n_part, -1)

    # fills vectors with info about particle loss
    lost_element = list(args.lost_element)
    lost_plane = list(args.lost_plane)
    lost_flag = list(args.lost_flag)

    return p_out, lost_flag, lost_element, lost_plane, lost_pos


@_interactive
def ring_pass(
    accelerator,
    particles,
    nr_turns=1,
    turn_by_turn=False,
    element_offset=0,
    parallel=False
):
    """Track particle(s) along a ring.

    Accepts one or multiple particles initial positions. In the latter case,
    a list of particles or a numpy 2D array (with particle as firts index)
    should be given as input; tracked particles positions at the end of
    the ring are output variables, as well as information on whether particles
    have been lost along the tracking and where they were lost.

    Args:
        accelerator (pyaccel.accelerator.Accelerator): accelerator object.
        particles (list|numpy.ndarray): initial 6D particle(s) position(s).
            Few examples
            ex.1: particles = [rx,px,ry,py,de,dl]
            ex.2: particles = numpy.array([rx,px,ry,py,de,dl])
            ex.3: particles = numpy.zeros((6, Np))
        nr_turns (int, optional): number of turns around ring to track each
            particle. Defaults to 1.
        turn_by_turn (bool, optional): parameter indicating whether turn by
            turn positions are to be returned. If False ringpass returns
            particles positions only at the end of the ring, at the last turn.
            If True, ringpass returns positions at the beginning of every turn
            (including the first) and at the end of the ring in the last
            turn. Defaults to False.
        element_offset (int, optional): element offset for tracking. tracking
            will start at the element with index 'element_offset'.
            Defaults to 0.
        parallel (bool, optional): whether to parallelize calculation or not.
            If an integer is passed that many processes will be used. If True,
            the number of processes will be determined automatically.
            Defaults to False.

    Returns:
        part_out (numpy.ndarray, (6, Np, N)): 6D position for each particle at
            end of ring. The structure of 'part_out' depends on inputs
            'particles' and 'turn_by_turn'. If 'turn_by_turn' is False then
            only tracked positions at the end 'nr_turns' are returned. There
            are still two possibilities for the structure of part_out,
            depending on 'particles':
            (1) if 'particles' is a single particle, 'part_out' will also be a
                unidimensional numpy array:
                    ex.:particles = [rx1,px1,ry1,py1,de1,dl1]
                        turn_by_turn = False
                        part_out = numpy.array([rx2,px2,ry2,py2,de2,dl2])
            (2) if 'particles' is either a numpy matrix with several particles
                then 'part_out' will be a 2D numpy whose
                first index selects the coordinate rx, px, ry, py, de or dl,
                in this order, and the second index selects a particular
                particle.
                    ex.: particles.shape == (6, Np)
                            turn_by_turn = False
                            part_out.shape == (6, Np))

            In case 'turn_by_turn' is True, 'part_out' will have tracked
            positions at the beggining of every turn and at the end of the
            last turn. The format of 'part_out' is
            (3) a numpy matrix, when 'particles' is a single particle. The
                first index of 'part_out' runs through coordinates rx, px, ry,
                py, de or dl and the second index runs through the turn number.
            (4) a 3D numpy, when 'particles' is the initial positions of many
                particles. The first index runs through coordinates, the
                second through particles and the third through turn number.

        loss_info (pyaccel.tracking.TrackingLossInfo): object with useful
            information about lost particles. Contains the following
            attributes:
            lost_flag (numpy.ndarray, dtype=bool): list of booleans indicating
                whether each particle was lost;
            lost_element (numpy.ndarray, dtype=int): list of element index
                where each particle was lost. If the particle survived the
                tracking through the line its corresponding element in this
                list is set to -1;
            lost_plane (numpy.ndarray, dtype=int): list of strings
                representing on what plane each particle was lost while being
                tracked. If the particle is not lost then its corresponding
                element in the list is set to None;
            lost_plane_str (numpy.ndarray, dtype=str): Same as previous but
                returns the string interpretation of the lost plane of each
                particles. Possible values are
                    ('no_plane', 'x', 'y', 'z', 'xy');
            lost_pos (numpy.ndarray, (6, Np)): 6D position vector of each lost
                particle at the moment they were lost. Position is set to NaN
                when particle is not lost;
            nr_particles (int): number of tracked particles;
            nr_lost_particles (int): number of lost particles;
            indices_lost_particles (numpy.ndarray, dtype=int): indices of lost
                particles.

    """
    # checks whether single or multiple particles, reformats particles
    p_in, *_ = _process_args(accelerator, particles, indices=None)

    if not parallel:
        out = _ring_pass(
            accelerator, p_in, nr_turns, turn_by_turn, element_offset
        )
        p_out, lost_flag, lost_turn, lost_element, lost_plane, lost_pos = out
    else:
        slcs = _get_slices_multiprocessing(parallel, p_in.shape[1])
        with _multiproc.Pool(processes=len(slcs)) as pool:
            res = []
            for slc in slcs:
                res.append(pool.apply_async(_ring_pass, (
                    accelerator, p_in[:, slc], nr_turns, turn_by_turn,
                    element_offset, True)))

            p_out, lost_pos = [], []
            lost_flag, lost_turn, lost_element, lost_plane = [], [], [], []
            for re_ in res:
                part_out, lflag, lturn, lelement, lplane, lpos = re_.get()
                p_out.append(part_out)
                lost_pos.append(lpos)
                lost_turn.extend(lturn)
                lost_element.extend(lelement)
                lost_plane.extend(lplane)
                lost_flag.extend(lflag)
        p_out = _np.concatenate(p_out, axis=1)
        lost_pos = _np.concatenate(lost_pos, axis=1)

    p_out = _np.squeeze(p_out)
    lost_pos = _np.squeeze(lost_pos)
    loss_info = TrackingLossInfo(
        lost_flag, lost_plane, lost_element, lost_pos, lost_turn=lost_turn
    )
    return p_out, loss_info


def _ring_pass(
    accelerator,
    p_in,
    nr_turns,
    turn_by_turn,
    element_offset,
    set_seed=False
):
    # static parameters of ringpass
    args = _trackcpp.RingPassArgs()
    args.nr_turns = int(nr_turns)
    args.turn_by_turn = bool(turn_by_turn)
    args.element_offset = int(element_offset)

    p_in = p_in.copy()
    n_part = p_in.shape[1]
    if bool(turn_by_turn):
        p_out = _np.zeros((6, n_part*(nr_turns+1)), dtype=float)
    else:
        p_out = _np.zeros((6, n_part), dtype=float)

    # re-seed pseudo-random generator
    if set_seed:
        _set_random_seed()

    # tracking
    _trackcpp.track_ringpass_wrapper(
        accelerator.trackcpp_acc, p_in, p_out, args
    )

    p_out = p_out.reshape(6, n_part, -1)
    # After tracking, the input vector contains the lost positions.
    p_lost = p_in

    # fills vectors with info about particle loss
    lost_turn = list(args.lost_turn)
    lost_element = list(args.lost_element)
    lost_plane = list(args.lost_plane)
    lost_flag = list(args.lost_flag)

    return p_out, lost_flag, lost_turn, lost_element, lost_plane, p_lost


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
        raise TrackingError(_trackcpp.string_error_messages[ret])

    closed_orbit = _CppDoublePosVector24Numpy(_closed_orbit)
    return closed_orbit[:, indices]


@_interactive
def find_orbit6(accelerator, indices=None, fixed_point_guess=None):
    """Calculate 6D closed orbit of accelerator and return it.

    Accepts an optional list of indices of ring elements where closed orbit
    coordinates are to be returned. If this argument is not passed, closed
    orbit positions are returned at the start of the first element. In
    addition a guess fixed point at the entrance of the ring may be provided.

    The radiation_on property will be temporarily set to "damping" to perform
    this calculation, regardless the initial radiation state.

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

    # The orbit can't be found when quantum excitation is on.
    rad_stt = accelerator.radiation_on
    if rad_stt == accelerator.RadiationStates.Full:
        accelerator.radiation_on = accelerator.RadiationStates.Damping

    if fixed_point_guess is None:
        fixed_point_guess = _trackcpp.CppDoublePos()
    else:
        fixed_point_guess = _Numpy2CppDoublePos(fixed_point_guess)

    _closed_orbit = _trackcpp.CppDoublePosVector()

    ret = _trackcpp.track_findorbit6(
        accelerator.trackcpp_acc, _closed_orbit, fixed_point_guess)

    accelerator.radiation_on = rad_stt

    if ret > 0:
        raise TrackingError(_trackcpp.string_error_messages[ret])

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
        raise TrackingError('The radiation is on but the cavity is off')
    else:
        return find_orbit6(
            accelerator, indices=indices, fixed_point_guess=fixed_point_guess
        )


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
            raise TrackingError(_trackcpp.string_error_messages[ret])
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
        raise TrackingError(_trackcpp.string_error_messages[ret])

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
            raise TrackingError(_trackcpp.string_error_messages[ret])
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
        raise TrackingError(_trackcpp.string_error_messages[ret])

    if indices is None:
        return m44
    return m44, cumul_trans_matrices


# ------ Auxiliary methods -------


def _set_random_seed():
    _trackcpp.set_random_seed_with_random_device()


def _get_slices_multiprocessing(parallel, nparticles):
    nrproc = _multiproc.cpu_count() - 3
    nrproc = nrproc if parallel is True else parallel
    nrproc = max(nrproc, 1)
    nrproc = min(nrproc, nparticles)

    np_proc = (nparticles // nrproc)*_np.ones(nrproc, dtype=int)
    np_proc[:(nparticles % nrproc)] += 1
    parts_proc = _np.r_[0, _np.cumsum(np_proc)]
    return [slice(parts_proc[i], parts_proc[i+1]) for i in range(nrproc)]


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


def _CppDoublePosVector2Numpy(poss):
    if not isinstance(poss, _trackcpp.CppDoublePosVector):
        raise TrackingError('invalid positions argument')

    poss_out = _np.zeros((6, poss.size()))
    for i, pos in enumerate(poss):
        poss_out[:, i] = _CppDoublePos2Numpy(pos)
    return poss_out


def _CppDoublePosVector24Numpy(poss):
    if not isinstance(poss, _trackcpp.CppDoublePosVector):
        raise TrackingError('invalid positions argument')

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
            raise TrackingError('dimension argument must be 4d or 6d.')
        posdim = 4 if dim == '4d' else 6
        if len(pos.shape) == 1:
            pos = _np.array(pos, ndmin=2).T
        elif len(pos.shape) > 2 or pos.shape[0] != posdim:
            raise TrackingError('invalid position argument.')
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
            raise TrackingError("invalid value for 'indices'")
    elif isinstance(indices, (list, tuple, _np.ndarray)):
        try:
            indices = _np.array(indices, dtype=int)
        except ValueError as err:
            raise TrackingError("invalid value for 'indices'") from err
        if len(indices.shape) > 1:
            raise TrackingError("invalid value for 'indices'")
    else:
        raise TrackingError("invalid value for 'indices'")
    return indices
