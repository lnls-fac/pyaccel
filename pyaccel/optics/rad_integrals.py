"""."""

import math as _math
import numpy as _np

import mathphys as _mp

from .. import lattice as _lattice
from .. import accelerator as _accelerator

from .twiss import calc_twiss as _calc_twiss
from .miscellaneous import get_rf_voltage as _get_rf_voltage, \
    get_revolution_frequency as _get_revolution_frequency, \
    get_curlyh as _get_curlyh, get_mcf as _get_mcf
from .eq_params import EqParamsXYModes as _EqParamsXYModes


class EqParamsFromRadIntegrals:
    """."""

    PARAMETERS = _EqParamsXYModes.PARAMETERS.union({
        'energy_offset', 'twiss',
        'I1x', 'I2', 'I3', 'I3a', 'I4x', 'I5x', 'I6x',
        'I1y', 'I4y', 'I5y', 'I6y'
        })

    def __init__(self, accelerator, energy_offset=0.0):
        """."""
        self._acc = _accelerator.Accelerator()
        self._energy_offset = energy_offset
        self._m66 = None
        self._twi = None
        self._alpha = 0.0
        self._integralsx = _np.zeros(6)
        self._integralsy = _np.zeros(6)
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

        ints = 'I1x,I4x,I5x,I6x'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'I1y,I4y,I5y,I6y'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'I2,I3,I3a'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        rst += _EqParamsXYModes.eqparam_to_string(self)
        return rst

    @property
    def accelerator(self):
        """."""
        return self._acc

    @accelerator.setter
    def accelerator(self, acc):
        if isinstance(acc, _accelerator.Accelerator):
            self._acc = acc
            self._calc_radiation_integrals()

    @property
    def energy_offset(self):
        """."""
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, value):
        self._energy_offset = float(value)
        self._calc_radiation_integrals()

    @property
    def twiss(self):
        """."""
        return self._twi

    @property
    def m66(self):
        """."""
        return self._m66

    @property
    def I1x(self):
        """."""
        return self._integralsx[0]

    @property
    def I2(self):
        """I2 is the same for x and y."""
        return self._integralsx[1]

    @property
    def I3(self):
        """I3 is the same for x and y."""
        return self._integralsx[2]

    @property
    def I3a(self):
        """I3a is the same for x and y."""
        return self._integralsx[3]

    @property
    def I4x(self):
        """."""
        return self._integralsx[4]

    @property
    def I5x(self):
        """."""
        return self._integralsx[5]

    @property
    def I6x(self):
        """."""
        return self._integralsx[6]

    @property
    def I1y(self):
        """."""
        return self._integralsy[0]

    @property
    def I4y(self):
        """."""
        return self._integralsy[4]

    @property
    def I5y(self):
        """."""
        return self._integralsy[5]

    @property
    def I6y(self):
        """."""
        return self._integralsy[6]

    @property
    def Jx(self):
        """."""
        return 1.0 - self.I4x/self.I2

    @property
    def Jy(self):
        """."""
        return 1.0 - self.I4y/self.I2

    @property
    def Je(self):
        """."""
        return 2.0 + (self.I4x + self.I4y)/self.I2

    @property
    def alphax(self):
        """."""
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # [GeV]
        E0 *= (1 + self._energy_offset)
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jx / leng

    @property
    def alphay(self):
        """."""
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # [GeV]
        E0 *= (1 + self._energy_offset)
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jy / leng

    @property
    def alphae(self):
        """."""
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # [GeV]
        E0 *= (1 + self._energy_offset)
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Je / leng

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
    def espread0(self):
        """."""
        Cq = _mp.constants.Cq
        gamma = self._acc.gamma_factor
        gamma *= (1 + self._energy_offset)
        return _math.sqrt(
            Cq * gamma**2 * self.I3 / (2*self.I2 + self.I4x + self.I4y))

    @property
    def emitx(self):
        """."""
        Cq = _mp.constants.Cq
        gamma = self._acc.gamma_factor
        gamma *= (1 + self._energy_offset)
        return Cq * gamma**2 * self.I5x / (self.Jx*self.I2)

    @property
    def emity(self):
        """."""
        Cq = _mp.constants.Cq
        gamma = self._acc.gamma_factor
        gamma *= (1 + self._energy_offset)
        return Cq * gamma**2 * self.I5y / (self.Jy*self.I2)

    @property
    def emit0(self):
        """."""
        return self.emitx + self.emity

    @property
    def U0(self):
        """Return U0 [eV]."""
        res = _EqParamsXYModes.calc_U0(self._acc, self.I2, self.energy_offset)
        return res

    @property
    def overvoltage(self):
        """."""
        v_cav = _get_rf_voltage(self._acc)
        return v_cav/self.U0

    @property
    def syncphase(self):
        """."""
        res = _EqParamsXYModes.calc_syncphase(
            self.overvoltage)
        return res

    @property
    def alpha(self):
        """."""
        return self._alpha

    @property
    def etac(self):
        """."""
        gamma = self._acc.gamma_factor
        gamma *= (1 + self._energy_offset)
        return 1/(gamma*gamma) - self.alpha

    @property
    def synctune(self):
        """."""
        E0 = self._acc.energy
        E0 *= (1 + self._energy_offset)
        v_cav = _get_rf_voltage(self._acc)
        harmon = self._acc.harmonic_number
        return _math.sqrt(
            self.etac*harmon*v_cav*_math.cos(self.syncphase)/(2*_math.pi*E0))

    @property
    def bunlen(self):
        """."""
        vel = self._acc.velocity
        rev_freq = _get_revolution_frequency(self._acc)

        bunlen = vel * abs(self.etac) * self.espread0
        bunlen /= 2*_math.pi * self.synctune * rev_freq
        return bunlen

    @property
    def rf_acceptance(self):
        """."""
        res = _EqParamsXYModes.calc_rf_acceptance(
            self._acc, self.syncphase, self.overvoltage, self.etac)
        return res

    @property
    def sigma_rx(self):
        """."""
        emitx = self.emitx
        espread0 = self.espread0
        return _np.sqrt(emitx*self._twi.betax + (espread0*self._twi.etax)**2)

    @property
    def sigma_px(self):
        """."""
        emitx = self.emitx
        espread0 = self.espread0
        return _np.sqrt(emitx*self._twi.gammax + (espread0*self._twi.etapx)**2)

    @property
    def sigma_ry(self):
        """."""
        emity = self.emity
        espread0 = self.espread0
        return _np.sqrt(emity*self._twi.betay + (espread0*self._twi.etay)**2)

    @property
    def sigma_py(self):
        """."""
        emity = self.emity
        espread0 = self.espread0
        return _np.sqrt(emity*self._twi.gammay + (espread0*self._twi.etapy)**2)

    def as_dict(self):
        """."""
        dic = {par: getattr(self, par) for par in self.PARAMETERS}
        dic['energy'] = self.accelerator.energy
        return dic

    def _calc_radiation_integrals(self):
        """Calculate radiation integrals for periodic systems."""
        acc = self._acc
        twi, m66 = _calc_twiss(
            acc, indices='closed', energy_offset=self._energy_offset)
        self._twi = twi
        self._m66 = m66
        self._alpha = _get_mcf(acc, energy_offset=self._energy_offset)

        spos = _lattice.find_spos(acc, indices='closed')
        etax, etapx, betax, alphax = twi.etax, twi.etapx, twi.betax, twi.alphax
        etay, etapy, betay, alphay = twi.etay, twi.etapy, twi.betay, twi.alphay

        n = len(acc)
        angle, angle_in, angle_out, K = _np.zeros((4, n))
        for i in range(n):
            angle[i] = acc[i].angle
            angle_in[i] = acc[i].angle_in
            angle_out[i] = acc[i].angle_out
            K[i] = acc[i].K

        idx, *_ = _np.nonzero(angle)
        leng = spos[idx+1]-spos[idx]
        rho = leng/angle[idx]
        angle_in = angle_in[idx]
        angle_out = angle_out[idx]
        K = K[idx]
        etax_in, etax_out = etax[idx], etax[idx+1]
        etapx_in, etapx_out = etapx[idx], etapx[idx+1]
        betax_in, betax_out = betax[idx], betax[idx+1]
        alphax_in, alphax_out = alphax[idx], alphax[idx+1]

        etay_in, etay_out = etay[idx], etay[idx+1]
        etapy_in, etapy_out = etapy[idx], etapy[idx+1]
        betay_in, betay_out = betay[idx], betay[idx+1]
        alphay_in, alphay_out = alphay[idx], alphay[idx+1]

        Hx_in = _get_curlyh(betax_in, alphax_in, etax_in, etapx_in)
        Hx_out = _get_curlyh(betax_out, alphax_out, etax_in, etapx_out)

        Hy_in = _get_curlyh(betay_in, alphay_in, etay_in, etapy_in)
        Hy_out = _get_curlyh(betay_out, alphay_out, etay_in, etapy_out)

        etax_avg = (etax_in + etax_out) / 2
        etay_avg = (etay_in + etay_out) / 2
        Hx_avg = (Hx_in + Hx_out) / 2
        Hy_avg = (Hy_in + Hy_out) / 2
        rho2, rho3 = rho**2, rho**3
        rho3abs = _np.abs(rho3)

        integralsx = _np.zeros(7)
        integralsx[0] = _np.dot(etax_avg/rho, leng)
        integralsx[1] = _np.dot(1/rho2, leng)
        integralsx[2] = _np.dot(1/rho3abs, leng)
        integralsx[3] = _np.dot(1/rho3, leng)

        integralsx[4] = _np.dot(etax_avg/rho3 * (1+2*rho2*K), leng)
        # for general wedge magnets:
        integralsx[4] += sum((etax_in/rho2) * _np.tan(angle_in))
        integralsx[4] += sum((etax_out/rho2) * _np.tan(angle_out))

        integralsx[5] = _np.dot(Hx_avg / rho3abs, leng)
        integralsx[6] = _np.dot((K*etax_avg)**2, leng)

        self._integralsx = integralsx

        integralsy = _np.zeros(7)
        integralsy[0] = _np.dot(etay_avg/rho, leng)
        integralsy[1] = integralsx[1]
        integralsy[2] = integralsx[2]
        integralsy[3] = integralsx[3]

        integralsy[4] = _np.dot(etay_avg/rho3 * (1+2*rho2*K), leng)
        # for general wedge magnets:
        integralsy[4] += sum((etay_in/rho2) * _np.tan(angle_in))
        integralsy[4] += sum((etay_out/rho2) * _np.tan(angle_out))

        integralsy[5] = _np.dot(Hy_avg / rho3abs, leng)
        integralsy[6] = _np.dot((K*etay_avg)**2, leng)
        self._integralsy = integralsy