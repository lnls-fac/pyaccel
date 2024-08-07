"""."""
import math as _math
import numpy as _np
import scipy.linalg as _scylin

import mathphys as _mp
import trackcpp as _trackcpp

from .. import tracking as _tracking
from .. import accelerator as _accelerator
from ..utils import interactive as _interactive

from .miscellaneous import get_rf_voltage as _get_rf_voltage, \
    get_revolution_frequency as _get_revolution_frequency, \
    calc_rf_acceptance as _calc_rf_acceptance, calc_U0 as _calc_U0, \
    calc_syncphase as _calc_syncphase
from .eq_params import EqParamsNormalModes as _EqParamsNormalModes


class EqParamsFromBeamEnvelope:
    """Calculate equilibrium beam parameters from beam envelope matrix.

    It employs Ohmi formalism to do so:
        Ohmi, Hirata, Oide 'From the beam-envelope matrix to synchrotron
        radiation integrals', Phys.Rev.E  Vol.49 p.751 (1994).
        https://doi.org/10.1103/PhysRevE.49.751
    Other useful reference is:
        Chao, A. W. (1979). Evaluation of beam distribution parameters in
        an electron storage ring. Journal of Applied Physics, 50(1), 595.
        https://doi.org/10.1063/1.326070


    The normal modes properties are defined so that in the limit of zero
    coupling:
        Mode 1 --> Horizontal plane
        Mode 2 --> Vertical plane
        Mode 3 --> Longitudinal plane

    """

    PARAMETERS = _EqParamsNormalModes.PARAMETERS.union({})

    def __init__(self, accelerator, energy_offset=0.0):
        """."""
        self._acc = _accelerator.Accelerator()
        self._energy_offset = energy_offset
        self._m66 = None
        self._cumul_mat = _np.zeros((len(self._acc)+1, 6, 6), dtype=float)
        self._bdiff = _np.zeros((len(self._acc)+1, 6, 6), dtype=float)
        self._envelope = _np.zeros((len(self._acc)+1, 6, 6), dtype=float)
        self._alpha = 0.0
        self._emits = _np.zeros(3)
        self._alphas = _np.zeros(3)
        self._damping_numbers = _np.zeros(3)
        self._tunes = _np.zeros(3)
        self._fixed_point = None
        self.accelerator = accelerator

    def __str__(self):
        """."""
        return _EqParamsNormalModes.eqparam_to_string(self)

    @property
    def accelerator(self):
        """."""
        return self._acc

    @accelerator.setter
    def accelerator(self, acc):
        if isinstance(acc, _accelerator.Accelerator):
            self._acc = acc
            self._calc_envelope()

    @property
    def energy(self):
        """."""
        return self._acc.energy

    @property
    def energy_offset(self):
        """."""
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, value):
        self._energy_offset = float(value)
        self._calc_envelope()

    @property
    def cumul_trans_matrices(self):
        """."""
        return self._cumul_mat.copy()

    @property
    def m66(self):
        """."""
        return self._cumul_mat[-1].copy()

    @property
    def fixed_point(self):
        """."""
        return self._fixed_point.copy() if self._fixed_point is not None else None

    @property
    def envelopes(self):
        """."""
        return self._envelope.copy()

    @property
    def diffusion_matrices(self):
        """."""
        return self._bdiff.copy()

    @property
    def tune1(self):
        """Tune of mode 1.

        In the limit of zero coupling, this is the horizontal tune.

        """
        return self._tunes[0]

    @property
    def tune2(self):
        """Tune of mode 2.

        In the limit of zero coupling, this is the vertical tune.

        """
        return self._tunes[1]

    @property
    def tune3(self):
        """Tune of mode 3.

        In the limit of zero coupling, this is the longitudinal tune.

        """
        return self._tunes[2]

    @property
    def alpha1(self):
        """Stationary damping rate of mode 1.

        In the limit of zero coupling, this is the horizontal damping rate.

        """
        return self._alphas[0]

    @property
    def alpha2(self):
        """Stationary damping rate of mode 2.

        In the limit of zero coupling, this is the vertical damping rate.

        """
        return self._alphas[1]

    @property
    def alpha3(self):
        """Stationary damping rate of mode 3.

        In the limit of zero coupling, this is the longitudinal damping rate.

        """
        return self._alphas[2]

    @property
    def tau1(self):
        """Stationary damping time of mode 1.

        In the limit of zero coupling, this is the horizontal damping time.

        """
        return 1/self.alpha1

    @property
    def tau2(self):
        """Stationary damping time of mode 2.

        In the limit of zero coupling, this is the vertical damping time.

        """
        return 1/self.alpha2

    @property
    def tau3(self):
        """Stationary damping time of mode 3.

        In the limit of zero coupling, this is the longitudinal damping time.

        """
        return 1/self.alpha3

    @property
    def J1(self):
        """Stationary damping number of mode 1.

        In the limit of zero coupling, this is the horizontal damping number.

        """
        return self._damping_numbers[0]

    @property
    def J2(self):
        """Stationary damping number of mode 2.

        In the limit of zero coupling, this is the vertical damping number.

        """
        return self._damping_numbers[1]

    @property
    def J3(self):
        """Stationary damping number of mode 3.

        In the limit of zero coupling, this is the longitudinal damping number.

        """
        return self._damping_numbers[2]

    @property
    def emit1(self):
        """Stationary emittance of mode 1.

        In the limit of zero coupling, this is the horizontal emittance.

        """
        return self._emits[0]

    @property
    def emit2(self):
        """Stationary emittance of mode 2.

        In the limit of zero coupling, this is the vertical emittance.

        """
        return self._emits[1]

    @property
    def emit3(self):
        """Stationary emittance of mode 3.

        In the limit of zero coupling, this is the longitudinal emittance.

        """
        return self._emits[2]

    @property
    def espread0(self):
        """Stationary energy spread."""
        return _np.sqrt(self._envelope[0, 4, 4])

    @property
    def bunlen(self):
        """Stationary bunch length."""
        return _np.sqrt(self._envelope[0, 5, 5])

    @property
    def sigma_rx(self):
        """Stationary horizontal size."""
        return _np.sqrt(self._envelope[:, 0, 0])

    @property
    def sigma_px(self):
        """Stationary horizontal divergence."""
        return _np.sqrt(self._envelope[:, 1, 1])

    @property
    def sigma_ry(self):
        """Stationary vertical size."""
        return _np.sqrt(self._envelope[:, 2, 2])

    @property
    def sigma_py(self):
        """Stationary vertical divergence."""
        return _np.sqrt(self._envelope[:, 3, 3])

    @property
    def tilt_xyplane(self):
        """Stationary tilt angle of the beam major axis in relation to X.

        Calculated via equation 25 of
            Chao, A. W. (1979). Evaluation of beam distribution parameters in
            an electron storage ring. Journal of Applied Physics, 50(1), 595.
            https://doi.org/10.1063/1.326070

        The equation reads:
            tilt = 1/2 * arctan(2*<xy>/(<x^2>-<y^2>))
        Besides this base equation, we also took into consideration the
        possibility of angles larger than pi/4 os smaller than -pi/4.

        """
        xx = self._envelope[:, 0, 0]
        yy = self._envelope[:, 2, 2]
        xy = self._envelope[:, 0, 2]
        dxy = xx - yy
        angles = _np.zeros(xx.size, dtype=float)
        idx = _np.isclose(dxy, 0.0, atol=1e-16)
        angles[idx] = _np.pi/4 * _np.sign(xy[idx])
        angles[~idx] = _np.arctan(2*xy[~idx]/_np.abs(dxy[~idx])) / 2
        idx = dxy < 0
        angles[idx] = _np.pi/2 * _np.sign(xy[idx]) - angles[idx]
        return angles

    @property
    def U0(self):
        """."""
        res = _calc_U0(self._acc.energy, self.energy_offset, self._integral2)
        return res

    @property
    def overvoltage(self):
        """."""
        v_cav = _get_rf_voltage(self._acc)
        return v_cav/self.U0

    @property
    def syncphase(self):
        """."""
        res = _calc_syncphase(self.overvoltage)
        return res

    @property
    def etac(self):
        """."""
        vel = self._acc.velocity
        rev_freq = _get_revolution_frequency(self._acc)

        # It is possible to infer the slippage factor via the relation between
        # the energy spread and the bunch length
        etac = self.bunlen / self.espread0 / vel
        etac *= 2*_math.pi * self.tune3 * rev_freq

        # Assume momentum compaction is positive and we are above transition:
        etac *= -1
        return etac

    @property
    def alpha(self):
        """."""
        # get alpha from slippage factor:
        gamma = self._acc.gamma_factor
        gamma *= (1 + self._energy_offset)
        return 1/(gamma*gamma) - self.etac

    @property
    def rf_acceptance(self):
        """."""
        rf_voltage = _get_rf_voltage(self._acc)
        res = _calc_rf_acceptance(
            self._acc.energy, self.energy_offset, self._acc.harmonic_number,
            rf_voltage, self.overvoltage, self.etac)
        return res

    def as_dict(self):
        """."""
        dic = {par: getattr(self, par) for par in self.PARAMETERS}
        dic['energy'] = self.accelerator.energy
        return dic

    def _calc_envelope(self):
        self._envelope, self._cumul_mat, self._bdiff, self._fixed_point = \
            calc_beamenvelope(
                self._acc, full=True, energy_offset=self._energy_offset)

        m66 = self._cumul_mat[-1]
        # # To calculate the emittances along the whole ring uncomment the
        # # line below:
        # m66 = _np.linalg.solve(
        #     self._cumul_mat.transpose(0, 2, 1),
        #     (self._cumul_mat @ m66).transpose(0, 2, 1)).transpose(0, 2, 1)

        # Look at section  D.2 of the Ohmi paper to understand this part of the
        # code on how to get the emmitances:

        # The function numpy.linalg.eig returns the evecs matrix such that:
        #    evecs^-1 @ m66 @ evecs = np.diag(evals)
        evals, evecs = _np.linalg.eig(m66)
        # Notice that the transformation generated by matrix evecs is the
        # inverse of equation 62 of the Ohmi paper.
        # So we need to calculate the inverse of evecs:
        evecsi = _np.linalg.inv(evecs)
        evecsih = evecsi.swapaxes(-1, -2).conj()

        # Then, using equation 64, we have:
        env0r = evecsi @ self._envelope[0] @ evecsih
        emits = _np.diagonal(
            env0r, axis1=-1, axis2=-2).real.take([0, 2, 4], axis=-1)

        # NOTE: I don't understand why I have to divide the resulting
        # emittances by the norms of evecsi[i, :]:
        norm_evecsi = _np.linalg.norm(evecsi, axis=-1)
        emits /= norm_evecsi.take([0, 2, 4], axis=-1)

        # get tunes and damping rates from one turn matrix
        trc = (evals[::2] + evals[1::2]).real
        dff = (evals[::2] - evals[1::2]).imag
        mus = _np.arctan2(dff, trc)
        alphas = trc / _np.cos(mus) / 2
        alphas = -_np.log(alphas) * _get_revolution_frequency(self._acc)

        # We have conventioned that in the limit of zero coupling mode 1 is
        # related to the horizontal plane, mode 2 is related to the vertical
        # plane and mode 3 is related to the longitunidal plane.
        # Since we know that in this limit the longitudinal emittance is
        # always the largest, the horizontal emittance is the second largest
        # and the vertical is the smallest, we order them in ascending order
        # and then define the modes ordering according the this convention:
        idx = _np.argsort(emits)
        idx = idx[[1, 0, 2]]  # from [V, H, L] to [H, V, L]
        self._alphas = alphas[idx]
        self._tunes = mus[idx] / 2 / _np.pi
        self._emits = emits[idx]

        # we know the damping numbers must sum to 4
        fac = _np.sum(self._alphas) / 4
        self._damping_numbers = self._alphas / fac

        # we can also extract the value of the second integral:
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # [GeV]
        E0 *= (1 + self._energy_offset)
        leng = self._acc.length
        self._integral2 = fac / (Ca * E0**3 / leng)


@_interactive
def calc_beamenvelope(
        accelerator, fixed_point=None, indices='closed', energy_offset=0.0,
        cumul_trans_matrices=None, init_env=None, full=False):
    """Calculate equilibrium beam envelope matrix or transport initial one.

    It employs Ohmi formalism to do so:
        Ohmi, Hirata, Oide 'From the beam-envelope matrix to synchrotron
        radiation integrals', Phys.Rev.E  Vol.49 p.751 (1994).
        https://doi.org/10.1103/PhysRevE.49.751

    Keyword arguments:
    accelerator : Accelerator object. Only non-optional argument.

    fixed_point : 6D position at the start of first element. It might be the
      fixed point of the one turn map or an arbitrary initial condition.

    indices : may be a (list,tuple, numpy.ndarray) of element indices where
      closed orbit data is to be returned or a string:
        'open'  : return the closed orbit at the entrance of all elements.
        'closed' : equal 'open' plus the orbit at the end of the last element.
      If indices is None the envelope is returned only at the entrance of
      the first element.

    energy_offset : float denoting the energy deviation (ignored if
      fixed_point is not None).

    cumul_trans_matrices : cumulated transfer matrices for all elements of the
      ring. Must include matrix at the end of the last element. If not passed
      or has the wrong shape it will be calculated internally.
      CAUTION: In case init_env is not passed and equilibrium solution is to
      be found, it must be calculated with cavity and radiation on.

    init_env: initial envelope matrix to be transported. In case it is not
      provided, the equilibrium solution will be returned.

    full: whether to return the envelope matrices as well as the the
      cumulative transfer matrices, the diffusion matrix and the fixed point.
      Defaults to False, in which case only the envelope is returned.

    Returns:
    envelope, if `full==False` -- rank-3 numpy array with shape
      (len(indices), 6, 6). Of the beam envelope matrices at the desired
      indices.

    envelope, cum_mat, bdiffs & fixed_point, if `full==True` -- the beam
      envelope matrices, the cumulated transfer matrices, and the diffusion
      matrices at the desired indices (rank-3 numpy arrays with shape
      (len(indices), 6, 6)); and the 6D position at the start of first element.
    """
    indices = _tracking._process_indices(accelerator, indices)

    rad_stt = accelerator.radiation_on
    cav_stt = accelerator.cavity_on
    accelerator.radiation_on = 'damping'
    accelerator.cavity_on = True

    if fixed_point is None:
        fixed_point = _tracking.find_orbit(
            accelerator, energy_offset=energy_offset)

    cum_mat = cumul_trans_matrices
    if cum_mat is None or cum_mat.shape[0] != len(accelerator)+1:
        _, cum_mat = _tracking.find_m66(
            accelerator, indices='closed', fixed_point=fixed_point)

    # perform: M(i, i) = M(0, i+1) @ M(0, i)^-1
    mat_ele = _np.linalg.solve(
        cum_mat[:-1].transpose((0, 2, 1)), cum_mat[1:].transpose((0, 2, 1)))
    mat_ele = mat_ele.transpose((0, 2, 1))
    mat_ele = mat_ele.copy()

    fixed_point = _tracking._Numpy2CppDoublePos(fixed_point)
    bdiffs = _np.zeros((len(accelerator)+1, 6, 6), dtype=float)
    _trackcpp.track_diffusionmatrix_wrapper(
        accelerator.trackcpp_acc, fixed_point, mat_ele, bdiffs)
    fixed_point = _tracking._CppDoublePos2Numpy(fixed_point)

    if init_env is None:
        # ------------------------------------------------------------
        # Equation for the moment matrix env is
        #        env = m66 @ env @ m66' + bcum;
        # We rewrite it in the form of the Sylvester equation:
        #        m66i @ env + env @ m66t = bcumi
        # where
        #        m66i =  inv(m66)
        #        m66t = -m66'
        #        bcumi = -m66i @ bcum
        # ------------------------------------------------------------
        m66 = cum_mat[-1]
        m66i = _np.linalg.inv(m66)
        m66t = -m66.T
        bcumi = _np.linalg.solve(m66, bdiffs[-1])
        # Envelope matrix at the ring entrance
        init_env = _scylin.solve_sylvester(m66i, m66t, bcumi)
        # Assert init_env is symmetric
        init_env += init_env.T
        init_env /= 2

    envelopes = _np.zeros((len(accelerator)+1, 6, 6), dtype=float)
    for i in range(envelopes.shape[0]):
        envelopes[i] = _sandwich_matrix(cum_mat[i], init_env) + bdiffs[i]
    # Assert envelopes are symmetric
    envelopes += envelopes.transpose(0, 2, 1)
    envelopes /= 2

    accelerator.radiation_on = rad_stt
    accelerator.cavity_on = cav_stt

    if not full:
        return envelopes[indices]
    return envelopes[indices], cum_mat[indices], bdiffs[indices], fixed_point


def _sandwich_matrix(mat1, mat2):
    """."""
    # return mat1 @ mat2 @ mat1.swapaxes(-1, -2)
    return _np.dot(mat1, _np.dot(mat2, mat1.T))
