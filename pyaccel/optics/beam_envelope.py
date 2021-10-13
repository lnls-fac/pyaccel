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
    get_revolution_frequency as _get_revolution_frequency


class EqParamsFromBeamEnvelope:
    """."""

    LONG = 2
    HORI = 1
    VERT = 0

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
        self.accelerator = accelerator

    def __str__(self):
        """."""
        rst = ''
        fmti = '{:32s}: '
        fmtr = '{:33s}: '
        fmtn = '{:.4g}'

        fmte = fmtr + fmtn
        rst += fmte.format('\nEnergy [GeV]', self.accelerator.energy*1e-9)
        rst += fmte.format('\nEnergy Deviation [%]', self.energy_offset*100)

        ints = 'Jx,Jy,Je'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'taux,tauy,taue'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [ms]')
        rst += ', '.join([fmtn.format(1000*getattr(self, x)) for x in ints])

        ints = 'alphax,alphay,alphae'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [Hz]')
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'tunex,tuney'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [Hz]')
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        rst += fmte.format('\nmomentum compaction x 1e4', self.alpha*1e4)
        rst += fmte.format('\nenergy loss [keV]', self.U0/1000)
        rst += fmte.format('\novervoltage', self.overvoltage)
        rst += fmte.format('\nsync phase [°]', self.syncphase*180/_math.pi)
        rst += fmte.format('\nsync tune', self.synctune)
        rst += fmte.format('\nhorizontal emittance [nm.rad]', self.emitx*1e9)
        rst += fmte.format('\nvertical emittance [pm.rad]', self.emity*1e12)
        rst += fmte.format('\nnatural emittance [nm.rad]', self.emit0*1e9)
        rst += fmte.format('\nnatural espread [%]', self.espread0*100)
        rst += fmte.format('\nbunch length [mm]', self.bunlen*1000)
        rst += fmte.format('\nRF energy accep. [%]', self.rf_acceptance*100)
        return rst

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
    def envelopes(self):
        """."""
        return self._envelope.copy()

    @property
    def diffusion_matrices(self):
        """."""
        return self._bdiff.copy()

    @property
    def tunex(self):
        """."""
        return self._tunes[self.HORI]

    @property
    def tuney(self):
        """."""
        return self._tunes[self.VERT]

    @property
    def synctune(self):
        """."""
        return self._tunes[self.LONG]

    @property
    def alphax(self):
        """."""
        return self._alphas[self.HORI]

    @property
    def alphay(self):
        """."""
        return self._alphas[self.VERT]

    @property
    def alphae(self):
        """."""
        return self._alphas[self.LONG]

    @property
    def taux(self):
        """."""
        return 1/self.alphax

    @property
    def tauy(self):
        """."""
        return 1/self.alphay

    @property
    def taue(self):
        """."""
        return 1/self.alphae

    @property
    def Jx(self):
        """."""
        return self._damping_numbers[self.HORI]

    @property
    def Jy(self):
        """."""
        return self._damping_numbers[self.VERT]

    @property
    def Je(self):
        """."""
        return self._damping_numbers[self.LONG]

    @property
    def espread0(self):
        """."""
        return _np.sqrt(self._envelope[0, 4, 4])

    @property
    def bunlen(self):
        """."""
        return _np.sqrt(self._envelope[0, 5, 5])

    @property
    def emitl(self):
        """."""
        return self._emits[2]

    @property
    def emitx(self):
        """."""
        return self._emits[1]

    @property
    def emity(self):
        """."""
        return self._emits[0]

    @property
    def emit0(self):
        """."""
        return _np.sum(self._emits[:-1])

    @property
    def sigma_rx(self):
        """."""
        return _np.sqrt(self._envelope[:, 0, 0])

    @property
    def sigma_px(self):
        """."""
        return _np.sqrt(self._envelope[:, 1, 1])

    @property
    def sigma_ry(self):
        """."""
        return _np.sqrt(self._envelope[:, 2, 2])

    @property
    def sigma_py(self):
        """."""
        return _np.sqrt(self._envelope[:, 3, 3])

    @property
    def U0(self):
        """."""
        E0 = self._acc.energy / 1e9  # in GeV
        rad_cgamma = _mp.constants.rad_cgamma
        return rad_cgamma/(2*_math.pi) * E0**4 * self._integral2 * 1e9  # in eV

    @property
    def overvoltage(self):
        """."""
        v_cav = _get_rf_voltage(self._acc)
        return v_cav/self.U0

    @property
    def syncphase(self):
        """."""
        return _math.pi - _math.asin(1/self.overvoltage)

    @property
    def etac(self):
        """."""
        vel = self._acc.velocity
        rev_freq = _get_revolution_frequency(self._acc)

        # It is possible to infer the slippage factor via the relation between
        # the energy spread and the bunch length
        etac = self.bunlen / self.espread0 / vel
        etac *= 2*_math.pi * self.synctune * rev_freq

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
        E0 = self._acc.energy
        sph = self.syncphase
        V = _get_rf_voltage(self._acc)
        ov = self.overvoltage
        h = self._acc.harmonic_number
        etac = self.etac

        eaccep2 = V * _math.sin(sph) / (_math.pi*h*abs(etac)*E0)
        eaccep2 *= 2 * (_math.sqrt(ov**2 - 1.0) - _math.acos(1.0/ov))
        return _math.sqrt(eaccep2)

    def as_dict(self):
        """."""
        pars = {
            'Jx', 'Jy', 'Je',
            'alphax', 'alphay', 'alphae',
            'taux', 'tauy', 'taue',
            'espread0',
            'emitx', 'emity', 'emit0',
            'bunlen',
            'U0', 'overvoltage', 'syncphase', 'synctune',
            'alpha', 'etac', 'rf_acceptance',
            }
        dic = {par: getattr(self, par) for par in pars}
        dic['energy'] = self.accelerator.energy
        return dic

    def _calc_envelope(self):
        self._envelope, self._cumul_mat, self._bdiff, self._fixed_point = \
            calc_beamenvelope(
                self._acc, full=True, energy_offset=self._energy_offset)

        m66 = self._cumul_mat[-1]

        # Look at section  D.2 of the Ohmi paper to understand this part of the
        # code on how to get the emmitances:
        evals, evecs = _np.linalg.eig(m66)
        # evecsh = evecs.swapaxes(-1, -2).conj()
        evecsi = _np.linalg.inv(evecs)
        evecsih = evecsi.swapaxes(-1, -2).conj()
        env0r = evecsi @ self._envelope[0] @ evecsih
        emits = _np.diagonal(env0r, axis1=-1, axis2=-2).real[::2].copy()
        emits /= _np.linalg.norm(evecsi, axis=-1)[::2]
        # # To calculate the emittances along the whole ring use this code:
        # m66i = self._cumul_mat @ m66 @ _np.linalg.inv(self._cumul_mat)
        # _, evecs = np.linalg.eig(m66i)
        # evecsi = np.linalg.inv(evecs)
        # evecsih = evecsi.swapaxes(-1, -2).conj()
        # env0r = evecsi @ self._envelope @ evecsih
        # emits = np.diagonal(env0r, axis1=-1, axis2=-2).real[:, ::2] * 1e12
        # emits /= np.linalg.norm(evecsi, axis=-1)[:, ::2]

        # get tunes and damping rates from one turn matrix
        trc = (evals[::2] + evals[1::2]).real
        dff = (evals[::2] - evals[1::2]).imag
        mus = _np.arctan2(dff, trc)
        alphas = trc / _np.cos(mus) / 2
        alphas = -_np.log(alphas) * _get_revolution_frequency(self._acc)

        # The longitudinal emittance is the largest one, then comes the
        # horizontal and latter the vertical:
        idx = _np.argsort(emits)
        self._alphas = alphas[idx]
        self._tunes = mus[idx] / 2 / _np.pi
        self._emits = emits[idx]

        # idcs = _np.r_[2*idx, 2*idx+1]
        # sig = env0r[:, idcs][idcs, :][:4, :4]
        # trans_evecs = evecs[:, idcs][idcs, :][:4, :4]
        # trans_evecsi = evecsi[:, idcs][idcs, :][:4, :4]

        # print(evecs @ evecsi)

        # trans_evecsh = evecsh[:, idcs][idcs, :][:4, :4]
        # sig = trans_evecs @ sig @ trans_evecsh
        # emity = _np.sqrt(_np.linalg.det(sig[:2, :2]).real)
        # emitx = _np.sqrt(_np.linalg.det(sig[2:4, 2:4]).real)
        # print(emitx, emity)

        # we know the damping numbers must sum to 4
        fac = _np.sum(self._alphas) / 4
        self._damping_numbers = self._alphas / fac

        # we can also extract the value of the second integral:
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        E0 *= (1 + self._energy_offset)
        leng = self._acc.length
        self._integral2 = fac / (Ca * E0**3 / leng)


@_interactive
def calc_beamenvelope(
        accelerator, fixed_point=None, indices='closed', energy_offset=0.0,
        cumul_trans_matrices=None, init_env=None, full=False):
    """Calculate equilibrium beam envelope matrix or transport initial one.

    It employs Ohmi formalism to do so:
        Ohmi, Kirata, Oide 'From the beam-envelope matrix to synchrotron
        radiation integrals', Phys.Rev.E  Vol.49 p.751 (1994)

    Keyword arguments:
    accelerator : Accelerator object. Only non-optional argument.

    fixed_point : 6D position at the start of first element. I might be the
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
      rin. Must include matrix at the end of the last element. If not passed
      or has the wrong shape it will be calculated internally.
      CAUTION: In case init_env is not passed and equilibrium solution is to
      be found, it must be calculated with cavity and radiation on.

    init_env: initial envelope matrix to be transported. In case it is not
      provided, the equilibrium solution will be returned.

    Returns:
    envelope -- rank-3 numpy array with shape (len(indices), 6, 6). Of the
      beam envelope matrices at the desired indices.

    """
    indices = _tracking._process_indices(accelerator, indices)

    if fixed_point is None:
        fixed_point = _tracking.find_orbit(
            accelerator, energy_offset=energy_offset)

    cum_mat = cumul_trans_matrices
    if cum_mat is None or cum_mat.shape[0] != len(accelerator)+1:
        if init_env is None:
            rad_stt = accelerator.radiation_on
            cav_stt = accelerator.cavity_on
            accelerator.radiation_on = True
            accelerator.cavity_on = True

        _, cum_mat = _tracking.find_m66(
            accelerator, indices='closed', fixed_point=fixed_point)

        if init_env is None:
            accelerator.radiation_on = rad_stt
            accelerator.cavity_on = cav_stt

    # perform: M(i, i) = M(0, i+1) @ M(0, i)^-1
    mat_ele = _np.linalg.solve(
        cum_mat[:-1].transpose((0, 2, 1)), cum_mat[1:].transpose((0, 2, 1)))
    mat_ele = mat_ele.transpose((0, 2, 1))
    mat_ele = mat_ele.copy()

    fixed_point = _tracking._Numpy2CppDoublePos(fixed_point)
    bdiffs = _np.zeros((len(accelerator)+1, 6, 6), dtype=float)
    _trackcpp.track_diffusionmatrix_wrapper(
        accelerator.trackcpp_acc, fixed_point, mat_ele, bdiffs)

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

    envelopes = _np.zeros((len(accelerator)+1, 6, 6), dtype=float)
    for i in range(envelopes.shape[0]):
        envelopes[i] = _sandwich_matrix(cum_mat[i], init_env) + bdiffs[i]

    if not full:
        return envelopes[indices]
    return envelopes[indices], cum_mat[indices], bdiffs[indices], fixed_point


def _sandwich_matrix(mat1, mat2):
    """."""
    # return mat1 @ mat2 @ mat1.swapaxes(-1, -2)
    return _np.dot(mat1, _np.dot(mat2, mat1.T))
