"""Beam lifetime calculation."""

import os as _os
import importlib as _implib
from copy import deepcopy as _dcopy
import numpy as _np

from mathphys import constants as _cst, units as _u, \
    beam_optics as _beam

from . import optics as _optics

import scipy.special as _special
if _implib.util.find_spec('scipy'):
    import scipy.integrate as _integrate
else:
    _integrate = None


class Lifetime:
    """Class which calculates the lifetime for a given accelerator."""

    # Constant factors
    _MBAR_2_PASCAL = 1.0e-3 / _u.pascal_2_bar

    _D_TOUSCHEK_FILE = _os.path.join(
        _os.path.dirname(__file__), 'data', 'd_touschek.npz')

    _KSI_TABLE = None
    _D_TABLE = None

    def __init__(self, accelerator):
        """."""
        self._acc = accelerator
        self._eqpar = _optics.EqParamsFromBeamEnvelope(self._acc)
        self._twiss, *_ = _optics.calc_twiss(self._acc, indices='closed')
        res = _optics.calc_transverse_acceptance(self._acc, self._twiss)
        self._accepx_nom = _np.min(res[0])
        self._accepy_nom = _np.min(res[1])
        self._curr_per_bun = 100/864  # [mA]
        self._avg_pressure = 1e-9  # [mbar]
        self._atomic_number = 7
        self._temperature = 300  # [K]
        self._taux = self._tauy = self._taue = None
        self._emitx = self._emity = self._espread0 = self._bunlen = None
        self._accepx = self._accepy = self._accepen = None
        self.touschek_model = 'piwinski'

    @property
    def accelerator(self):
        """."""
        return self._acc

    @accelerator.setter
    def accelerator(self, val):
        self._eqpar = _optics.EqParamsFromBeamEnvelope(val)
        self._twiss, *_ = _optics.calc_twiss(val, indices='closed')
        res = _optics.calc_transverse_acceptance(val, self._twiss)
        self._accepx_nom = _np.min(res[0])
        self._accepy_nom = _np.min(res[1])
        self._acc = val

    @property
    def equi_params(self):
        """Equilibrium parameters."""
        return self._eqpar

    @property
    def twiss(self):
        """Twiss data."""
        return self._twiss

    @property
    def curr_per_bunch(self):
        """Return current per bunch [mA]."""
        return self._curr_per_bun

    @curr_per_bunch.setter
    def curr_per_bunch(self, val):
        self._curr_per_bun = float(val)

    @property
    def particles_per_bunch(self):
        """Particles per bunch."""
        return int(_beam.calc_number_of_electrons(
            self._acc.energy * _u.eV_2_GeV, self.curr_per_bunch,
            self._acc.length))

    @property
    def avg_pressure(self):
        """Average Pressure [mbar]."""
        return self._avg_pressure

    @avg_pressure.setter
    def avg_pressure(self, val):
        self._avg_pressure = float(val)

    @property
    def atomic_number(self):
        """Atomic number of residual gas."""
        return self._atomic_number

    @atomic_number.setter
    def atomic_number(self, val):
        self._atomic_number = int(val)

    @property
    def temperature(self):
        """Average Temperature of residual gas [K]."""
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = float(val)

    @property
    def emitx(self):
        """Horizontal Emittance [m.rad]."""
        if self._emitx is not None:
            return self._emitx
        return self._eqpar.emitx

    @emitx.setter
    def emitx(self, val):
        self._emitx = float(val)

    @property
    def emity(self):
        """Vertical Emittance [m.rad]."""
        if self._emity is not None:
            return self._emity
        return self._eqpar.emity

    @emity.setter
    def emity(self, val):
        self._emity = float(val)

    @property
    def espread0(self):
        """Relative energy spread."""
        if self._espread0 is not None:
            return self._espread0
        return self._eqpar.espread0

    @espread0.setter
    def espread0(self, val):
        self._espread0 = float(val)

    @property
    def bunlen(self):
        """Bunch length [m]."""
        if self._bunlen is not None:
            return self._bunlen
        return self._eqpar.bunlen

    @bunlen.setter
    def bunlen(self, val):
        self._bunlen = float(val)

    @property
    def taux(self):
        """Horizontal damping Time [s]."""
        if self._taux is not None:
            return self._taux
        return self._eqpar.taux

    @taux.setter
    def taux(self, val):
        self._taux = float(val)

    @property
    def tauy(self):
        """Vertical damping Time [s]."""
        if self._tauy is not None:
            return self._tauy
        return self._eqpar.tauy

    @tauy.setter
    def tauy(self, val):
        self._tauy = float(val)

    @property
    def taue(self):
        """Longitudinal damping Time [s]."""
        if self._taue is not None:
            return self._taue
        return self._eqpar.taue

    @taue.setter
    def taue(self, val):
        self._taue = float(val)

    @property
    def accepen(self):
        """Longitudinal acceptance."""
        if self._accepen is not None:
            return self._accepen
        dic = dict()
        rf_accep = self._eqpar.rf_acceptance
        dic['spos'] = self._twiss.spos
        dic['accp'] = dic['spos']*0 + rf_accep
        dic['accn'] = dic['spos']*0 - rf_accep
        return dic

    @accepen.setter
    def accepen(self, val):
        if isinstance(val, dict):
            if {'spos', 'accp', 'accn'} - val.keys():
                raise KeyError(
                    "Dictionary must contain keys 'spos', 'accp', 'accn'")
            spos = val['spos']
            accp = val['accp']
            accn = val['accn']
        elif isinstance(val, (list, tuple, _np.ndarray)):
            spos = self._twiss.spos
            accp = spos*0.0 + val[1]
            accn = spos*0.0 + val[0]
        elif isinstance(val, (int, _np.int, float, _np.float)):
            spos = self._twiss.spos
            accp = spos*0.0 + val
            accn = spos*0.0 - val
        else:
            raise TypeError('Wrong value for energy acceptance')
        self._accepen = _dcopy(dict(spos=spos, accp=accp, accn=accn))

    @property
    def accepx(self):
        """Horizontal acceptance."""
        if self._accepx is not None:
            return self._accepx
        dic = dict()
        dic['spos'] = self._twiss.spos
        dic['acc'] = dic['spos']*0 + self._accepx_nom
        return dic

    @accepx.setter
    def accepx(self, val):
        if isinstance(val, dict):
            if {'spos', 'acc'} - val.keys():
                raise KeyError(
                    "Dictionary must contain keys 'spos', 'acc'")
            spos = val['spos']
            acc = val['acc']
        elif isinstance(val, (int, _np.int, float, _np.float)):
            spos = self._twiss.spos
            acc = spos*0.0 + val
        else:
            raise TypeError('Wrong value for energy acceptance')
        self._accepx = _dcopy(dict(spos=spos, acc=acc))

    @property
    def accepy(self):
        """Vertical acceptance."""
        if self._accepy is not None:
            return self._accepy
        dic = dict()
        dic['spos'] = self._twiss.spos
        dic['acc'] = dic['spos']*0 + self._accepy_nom
        return dic

    @accepy.setter
    def accepy(self, val):
        if isinstance(val, dict):
            if {'spos', 'acc'} - val.keys():
                raise KeyError(
                    "Dictionary must contain keys 'spos', 'acc'")
            spos = val['spos']
            acc = val['acc']
        elif isinstance(val, (int, _np.int, float, _np.float)):
            spos = self._twiss.spos
            acc = spos*0.0 + val
        else:
            raise TypeError('Wrong value for energy acceptance')
        self._accepy = _dcopy(dict(spos=spos, acc=acc))

    @property
    def touschek_data(self):
        """Calculate loss rate due to Touschek beam lifetime.

        parameters used in calculation:

        emitx        = Horizontal emittance [m.rad]
        emity        = Vertical emittance [m.rad]
        energy       = Bunch energy [GeV]
        nr_part      = Number of electrons ber bunch
        espread      = relative energy spread,
        bunlen       = bunch length [m]
        accepen      = relative energy acceptance of the machine.

        twiss = pyaccel.TwissArray object or similar object with fields:
                spos, betax, betay, etax, etay, alphax, alphay, etapx, etapy

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
            volume   = volume of the beam along the ring [m^3]
        """
        self._load_touschek_integration_table()
        gamma = self._acc.gamma_factor
        beta = self._acc.beta_factor
        en_accep = self.accepen
        twiss = self._twiss
        emitx, emity = self.emitx, self.emity
        espread = self.espread0
        bunlen = self.bunlen
        nr_part = self.particles_per_bunch

        _, ind = _np.unique(twiss.spos, return_index=True)
        spos = en_accep['spos']
        accp = en_accep['accp']
        accn = en_accep['accn']

        # calcular o tempo de vida a cada 10 cm do anel:
        npoints = int((spos[-1] - spos[0])/0.1)
        s_calc = _np.linspace(spos[0], spos[-1], npoints)
        d_accp = _np.interp(s_calc, spos, accp)
        d_accn = _np.interp(s_calc, spos, -accn)

        # if momentum aperture is 0, set it to 1e-4:
        d_accp[d_accp == 0] = 1e-4
        d_accn[d_accn == 0] = 1e-4

        betax = _np.interp(s_calc, twiss.spos[ind], twiss.betax[ind])
        alphax = _np.interp(s_calc, twiss.spos[ind], twiss.alphax[ind])
        etax = _np.interp(s_calc, twiss.spos[ind], twiss.etax[ind])
        etaxl = _np.interp(s_calc, twiss.spos[ind], twiss.etapx[ind])

        betay = _np.interp(s_calc, twiss.spos[ind], twiss.betay[ind])
        alphay = _np.interp(s_calc, twiss.spos[ind], twiss.alphay[ind])
        etay = _np.interp(s_calc, twiss.spos[ind], twiss.etay[ind])
        etayl = _np.interp(s_calc, twiss.spos[ind], twiss.etapy[ind])

        # Tamanhos betatron do bunch
        sigxb2 = emitx * betax
        sigyb2 = emity * betay

        # Volume do bunch
        sigy = _np.sqrt(etay**2*espread**2 + betay*emity)
        sigx = _np.sqrt(etax**2*espread**2 + betax*emitx)
        vol = bunlen * sigx * sigy
        const = (_cst.electron_radius**2 * _cst.light_speed) / (8*_np.pi)

        if self.touschek_model == 'flat_beam':
            fator = betax*etaxl + alphax*etax
            a_var = 1 / (4*espread**2) + (etax**2 + fator**2) / (4*sigxb2)
            b_var = betax*fator / (2*sigxb2)
            c_var = betax**2 / (4*sigxb2) - b_var**2 / (4*a_var)

            # Limite de integração inferior
            ksip = (2*_np.sqrt(c_var)/gamma * d_accp)**2
            ksin = (2*_np.sqrt(c_var)/gamma * d_accn)**2

            # Interpola d_touschek
            d_pos = _np.interp(
                ksip, self._KSI_TABLE, self._D_TABLE, left=0.0, right=0.0)
            d_neg = _np.interp(
                ksin, self._KSI_TABLE, self._D_TABLE, left=0.0, right=0.0)

            # Tempo de vida touschek inverso
            ratep = const * nr_part/gamma**2 / d_accp**3 * d_pos / vol
            raten = const * nr_part/gamma**2 / d_accn**3 * d_neg / vol
            rate = (ratep+raten)/2
        elif self.touschek_model == 'piwinski':
            etaxtil2 = (alphax*etax + betax*etaxl)**2
            etaytil2 = (alphay*etay + betay*etayl)**2
            espread2 = espread*espread

            val1 = 1/espread2
            val2 = (etax*etax + etaxtil2)/(sigxb2)
            val3 = (etay*etay + etaytil2)/(sigyb2)
            sigh2 = 1/(val1 + val2 + val3)

            betagamma2 = (beta*gamma)**2

            cx_ = betax**2/sigxb2
            cy_ = betay**2/sigyb2
            b1_ = cx_*(1-sigh2*etaxtil2/sigxb2)
            b1_ += cy_*(1-sigh2*etaytil2/sigyb2)
            b1_ /= (2*betagamma2)

            ch_ = (sigx*sigy)**2 - (espread2*etax*etay)**2
            cb2 = sigh2/(betagamma2*emitx*emity)**2
            cb2 *= ch_/espread2
            b2_ = b1_**2 - cb2
            for idx in range(npoints):
                if b2_[idx] < 0:
                    if abs(b2_[idx]/b1_[idx]**2 < 1e-7):
                        print(f'B2^2 < 0 at {idx:04d}')
                    else:
                        print(f'Setting B2 to zero at {idx:04d}')
                        b2_[idx] = 0
            b2_ = _np.sqrt(b2_)

            taum_p = (beta*d_accp)**2
            taum_n = (beta*d_accn)**2

            rate = []
            for idx in range(npoints):
                f_int_p = self.f_integral_2_simps(
                    taum_p[idx], b1_[idx], b2_[idx])
                f_int_n = self.f_integral_2_simps(
                    taum_n[idx], b1_[idx], b2_[idx])
                rate_const = const * nr_part/gamma**2/bunlen
                rate_const /= _np.sqrt(ch_[idx])
                ratep = rate_const * f_int_p/taum_p[idx]
                raten = rate_const * f_int_n/taum_n[idx]
                rate.append((ratep + raten)/2)

        rate = _np.array(rate)
        # Tempo de vida touschek inverso médio
        avg_rate = _np.trapz(rate, x=s_calc) / (s_calc[-1] - s_calc[0])
        return dict(rate=rate, avg_rate=avg_rate, volume=vol, pos=s_calc)

    @staticmethod
    def f_function_arg_1(tau, taum, b1_, b2_):
        """."""
        ratio = tau/taum/(1+tau)
        arg = (2+1/tau)**2 * (ratio - 1)
        arg += 1 - _np.sqrt(1/ratio)
        arg -= 1/2/tau*(4 + 1/tau) * _np.log(ratio)
        arg *= _np.sqrt(ratio*taum)
        bessel = _np.exp(-b1_*tau)*_special.i0(b2_*tau)
        res = arg * bessel
        if _np.isnan(res).any() or _np.isinf(res).any():
            bessel = _np.exp(-(b1_-b2_)*tau)/_np.sqrt(2*_np.pi*tau*b2_)
            res = arg * bessel
        return res

    @staticmethod
    def f_function_arg_2(tau, taum, b1_, b2_):
        """."""
        tau = _np.tan(tau)**2
        ratio = tau/taum/(1+tau)
        arg = (2*tau+1)**2 * (ratio - 1)/tau
        arg += tau - _np.sqrt(tau*taum*(1+tau))
        arg -= (2+1/(2*tau))*_np.log(ratio)
        arg *= _np.sqrt(1+tau)
        bessel = _np.exp(-b1_*tau)*_special.i0(b2_*tau)
        res = arg * bessel
        if _np.isnan(res).any() or _np.isinf(res).any():
            bessel = _np.exp(-(b1_-b2_)*tau)/_np.sqrt(2*_np.pi*tau*b2_)
            res = arg * bessel
        return res

    @staticmethod
    def f_integral_1(taum, b1_, b2_):
        """."""
        lim = 1000
        f_int, _ = _integrate.quad(
            func=Lifetime.f_function_arg_1, a=taum, b=_np.inf,
            args=(taum, b1_, b2_), limit=lim)
        f_int *= _np.sqrt(_np.pi*(b1_**2-b2_**2))*taum
        return f_int

    @staticmethod
    def f_integral_2(taum, b1_, b2_):
        """."""
        lim = 1000
        kappam = _np.arctan(_np.sqrt(taum))
        f_int, _ = _integrate.quad(
            func=Lifetime.f_function_arg_2, a=kappam, b=_np.pi/2,
            args=(taum, b1_, b2_), limit=lim)
        f_int *= 2*_np.sqrt(_np.pi*(b1_**2-b2_**2))*taum
        return f_int

    @staticmethod
    def f_integral_2_simps(taum, b1_, b2_):
        """."""
        kappam = _np.arctan(_np.sqrt(taum))
        npts = int(3*70)
        dtau = (_np.pi/2-kappam)/npts
        tau = _np.linspace(kappam, _np.pi/2, npts+1)
        func = Lifetime.f_function_arg_2(tau, taum, b1_, b2_)

        # Simpson's 3/8 Rule - N must be mod(N, 3) = 0
        val1 = func[0:-1:3] + func[3::3]
        val2 = 3*(func[1::3] + func[2::3])
        f_int = 3*dtau/8 * _np.sum(val1 + val2)

        # Simpson's 1/3 Rule - N must be mod(N, 2) = 0
        # f_int = _np.sum(func[0:-1:2]+4*func[1::2] + func[2::2])
        # f_int *= dtau/3
        f_int *= 2*_np.sqrt(_np.pi*(b1_**2-b2_**2))*taum
        return f_int

    @property
    def lossrate_touschek(self):
        """Return Touschek loss rate [1/s]."""
        data = self.touschek_data
        return data['avg_rate']

    @property
    def elastic_data(self):
        """
        Calculate beam loss rate due to elastic scattering from residual gas.

        Parameters used in calculations:
        accepx, accepy = horizontal and vertical acceptances [m·rad]
        avg_pressure   = Residual gas pressure [mbar]
        atomic number  = Residual gas atomic number (default: 7)
        temperature    = Residual gas temperature [K] (default: 300)
        energy         = Beam energy [eV]
        twiss          = Twis parameters

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_x = self.accepx
        accep_y = self.accepy
        pressure = self.avg_pressure
        twiss = self._twiss
        energy = self._acc.energy
        beta = self._acc.beta_factor
        atomic_number = self.atomic_number
        temperature = self.temperature

        betax, betay = twiss.betax, twiss.betay
        energy_joule = energy / _u.joule_2_eV

        spos = twiss.spos
        _, idx = _np.unique(accep_x['spos'], return_index=True)
        _, idy = _np.unique(accep_y['spos'], return_index=True)
        accep_x = _np.interp(spos, accep_x['spos'][idx], accep_x['acc'][idx])
        accep_y = _np.interp(spos, accep_y['spos'][idy], accep_y['acc'][idy])

        thetax = _np.sqrt(accep_x/betax)
        thetay = _np.sqrt(accep_y/betay)
        ratio = thetay / thetax

        f_x = 2*_np.arctan(ratio) + _np.sin(2*_np.arctan(ratio))
        f_x *= pressure * self._MBAR_2_PASCAL * betax / accep_x
        f_y = _np.pi - 2*_np.arctan(ratio) + _np.sin(2*_np.arctan(ratio))
        f_y *= pressure * self._MBAR_2_PASCAL * betay / accep_y

        # Constant
        rate = _cst.light_speed * _cst.elementary_charge**4
        rate /= 4 * _np.pi**2 * _cst.vacuum_permitticity**2
        # Parameter dependent part
        rate *= atomic_number**2 * (f_x + f_y)
        rate /= beta * energy_joule**2
        rate /= temperature * _cst.boltzmann_constant

        avg_rate = _np.trapz(rate, spos) / (spos[-1]-spos[0])
        return dict(rate=rate, avg_rate=avg_rate, pos=spos)

    @property
    def lossrate_elastic(self):
        """Return elastic loss rate [1/s]."""
        data = self.elastic_data
        return data['avg_rate']

    @property
    def inelastic_data(self):
        """
        Calculate loss rate due to inelastic scattering beam lifetime.

        Parameters used in calculations:
        accepen       = Relative energy acceptance
        avg_pressure  = Residual gas pressure [mbar]
        atomic_number = Residual gas atomic number (default: 7)
        temperature   = [K] (default: 300)

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        en_accep = self.accepen
        pressure = self.avg_pressure
        atomic_number = self.atomic_number
        temperature = self.temperature

        spos = en_accep['spos']
        accp = en_accep['accp']
        accn = -en_accep['accn']

        rate = 32 * _cst.light_speed * _cst.electron_radius**2  # Constant
        rate /= 411 * _cst.boltzmann_constant * temperature  # Temperature
        rate *= atomic_number**2 * _np.log(183/atomic_number**(1/3))  # Z
        rate *= pressure * self._MBAR_2_PASCAL  # Pressure

        ratep = accp - _np.log(accp) - 5/8  # Eaccep
        raten = accn - _np.log(accn) - 5/8  # Eaccep
        rate *= (ratep + raten) / 2

        avg_rate = _np.trapz(rate, spos) / (spos[-1]-spos[0])
        return dict(rate=rate, avg_rate=avg_rate, pos=spos)

    @property
    def lossrate_inelastic(self):
        """Return inelastic loss rate [1/s]."""
        data = self.inelastic_data
        return data['avg_rate']

    @property
    def quantumx_data(self):
        """Beam loss rates in horizontal plane due to quantum excitation.

        Positional arguments:
        accepx   = horizontal acceptance [m·rad]
        emitx    = horizontal emittance [m·rad]
        taux     = horizontal damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_x = self.accepx
        emitx = self.emitx
        taux = self.taux

        spos = accep_x['spos']
        accep_x = accep_x['acc']

        ksi_x = accep_x / (2*emitx)
        rate = self._calc_quantum_loss_rate(ksi_x, taux)

        avg_rate = _np.trapz(rate, spos) / (spos[-1]-spos[0])
        return dict(rate=rate, avg_rate=avg_rate, pos=spos)

    @property
    def lossrate_quantumx(self):
        """Return quantum loss rate in horizontal plane [1/s]."""
        data = self.quantumx_data
        return data['avg_rate']

    @property
    def quantumy_data(self):
        """Beam loss rates in vertical plane due to quantum excitation.

        Positional arguments:
        accepy   = vertical acceptance [m·rad]
        emity    = vertical emittance [m·rad]
        tauy     = vertical damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_y = self.accepy
        emity = self.emity
        tauy = self.tauy

        spos = accep_y['spos']
        accep_y = accep_y['acc']

        ksi_y = accep_y / (2*emity)
        rate = self._calc_quantum_loss_rate(ksi_y, tauy)

        avg_rate = _np.trapz(rate, spos) / (spos[-1]-spos[0])
        return dict(rate=rate, avg_rate=avg_rate, pos=spos)

    @property
    def lossrate_quantumy(self):
        """Return quantum loss rate in vertical plane [1/s]."""
        data = self.quantumy_data
        return data['avg_rate']

    @property
    def quantume_data(self):
        """Beam loss rates in longitudinal plane due to quantum excitation.

        Positional arguments:
        accepen   = longitudinal acceptance [m·rad]
        espread0  = relative energy spread
        taue      = longitudinal damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        en_accep = self.accepen
        espread = self.espread0
        taue = self.taue

        spos = en_accep['spos']
        accp = en_accep['accp']
        accn = en_accep['accn']

        ratep = self._calc_quantum_loss_rate((accp/espread)**2 / 2, taue)
        raten = self._calc_quantum_loss_rate((accn/espread)**2 / 2, taue)
        rate = (ratep + raten) / 2

        avg_rate = _np.trapz(rate, spos) / (spos[-1]-spos[0])
        return dict(rate=rate, avg_rate=avg_rate, pos=spos)

    @property
    def lossrate_quantume(self):
        """Return quantum loss rate in longitudinal plane [1/s]."""
        data = self.quantume_data
        return data['avg_rate']

    @property
    def lossrate_quantum(self):
        """Return quantum loss rate [1/s]."""
        rate = self.lossrate_quantume
        rate += self.lossrate_quantumx
        rate += self.lossrate_quantumy
        return rate

    @property
    def lossrate_total(self):
        """Return total loss rate [1/s]."""
        rate = self.lossrate_elastic
        rate += self.lossrate_inelastic
        rate += self.lossrate_quantum
        rate += self.lossrate_touschek
        return rate

    @property
    def lifetime_touschek(self):
        """Return Touschek lifetime [s]."""
        loss = self.lossrate_touschek
        return 1 / loss if loss > 0 else _np.inf

    @property
    def lifetime_elastic(self):
        """Return elastic lifetime [s]."""
        loss = self.lossrate_elastic
        return 1 / loss if loss > 0 else _np.inf

    @property
    def lifetime_inelastic(self):
        """Return inelastic lifetime [s]."""
        loss = self.lossrate_inelastic
        return 1 / loss if loss > 0 else _np.inf

    @property
    def lifetime_quantum(self):
        """Return quandtum lifetime [s]."""
        loss = self.lossrate_quantum
        return 1 / loss if loss > 0 else _np.inf

    @property
    def lifetime_total(self):
        """Return total lifetime [s]."""
        loss = self.lossrate_total
        return 1 / loss if loss > 0 else _np.inf

    @classmethod
    def get_touschek_integration_table(cls, ksi_ini=None, ksi_end=None):
        """Return Touschek interpolation table."""
        if None in (ksi_ini, ksi_end):
            cls._load_touschek_integration_table()
        else:
            cls._calc_d_touschek_table(ksi_ini, ksi_end)
        return cls._KSI_TABLE, cls._D_TABLE

    # ----- private methods -----

    @staticmethod
    def _calc_quantum_loss_rate(ksi, tau):
        return 2*ksi*_np.exp(-ksi)/tau

    @classmethod
    def _load_touschek_integration_table(cls):
        if cls._KSI_TABLE is None or cls._D_TABLE is None:
            data = _np.load(cls._D_TOUSCHEK_FILE)
            cls._KSI_TABLE = data['ksi']
            cls._D_TABLE = data['d']

    @classmethod
    def _calc_d_touschek_table(cls, ksi_ini, ksi_end, npoints):
        if not _implib.util.find_spec('scipy'):
            raise NotImplementedError(
                'Scipy is needed for this calculation!')
        ksi_tab = _np.logspace(ksi_ini, ksi_end, npoints)
        d_tab = _np.zeros(ksi_tab.size)
        for i, ksi in enumerate(ksi_tab):
            d_tab[i] = cls._calc_d_touschek_scipy(ksi)
        cls._D_TABLE = d_tab
        cls._KSI_TABLE = ksi_tab

    @staticmethod
    def _calc_d_touschek_scipy(ksi):
        if _integrate is None:
            raise ImportError('scipy library not available')
        lim = 1000
        int1, _ = _integrate.quad(
            lambda x: _np.exp(-x)/x, ksi, _np.inf, limit=lim)
        int2, _ = _integrate.quad(
            lambda x: _np.exp(-x)*_np.log(x)/x, ksi, _np.inf, limit=lim)
        d_val = _np.sqrt(ksi)*(
            -1.5 * _np.exp(-ksi) +
            0.5 * (3*ksi - ksi*_np.log(ksi) + 2) * int1 +
            0.5 * ksi * int2
            )
        return d_val
