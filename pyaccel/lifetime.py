"""Beam lifetime calculation."""

import os as _os
import importlib as _implib
from copy import deepcopy as _dcopy
import numpy as _np

from mathphys import constants as _cst, units as _u, \
    beam_optics as _beam

from . import optics as _optics

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
        self._eqpar = _optics.EquilibriumParametersIntegrals(accelerator)
        res = _optics.get_transverse_acceptance(self._acc, self._eqpar.twiss)
        self._accepx_nom = _np.min(res[0])
        self._accepy_nom = _np.min(res[1])
        self._curr_per_bun = 100/864  # [mA]
        self._avg_pressure = 1e-9  # [mbar]
        self._coupling = 0.03
        self._atomic_number = 7
        self._temperature = 300  # [K]
        self._taux = self._tauy = self._taue = None
        self._emit0 = self._espread0 = self._bunlen = None
        self._accepx = self._accepy = self._accepen = None

    @property
    def accelerator(self):
        """."""
        return self._acc

    @accelerator.setter
    def accelerator(self, val):
        self._eqpar = _optics.EquilibriumParametersIntegrals(val)
        res = _optics.get_transverse_acceptance(val, self._eqpar.twiss)
        self._accepx_nom = _np.min(res[0])
        self._accepy_nom = _np.min(res[1])
        self._acc = val

    @property
    def equi_params(self):
        """Equilibrium parameters."""
        return self._eqpar

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
    def coupling(self):
        """Emittances ratio."""
        return self._coupling

    @coupling.setter
    def coupling(self, val):
        self._coupling = float(val)

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
    def emit0(self):
        """Transverse Emittance [m.rad]."""
        if self._emit0 is not None:
            return self._emit0
        return self._eqpar.emit0

    @emit0.setter
    def emit0(self, val):
        self._emit0 = float(val)

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
        dic['spos'] = self._eqpar.twiss.spos
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
            spos = self._eqpar.twiss.spos
            accp = spos*0.0 + val[1]
            accn = spos*0.0 + val[0]
        elif isinstance(val, (int, _np.int, float, _np.float)):
            spos = self._eqpar.twiss.spos
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
        dic['spos'] = self._eqpar.twiss.spos
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
            spos = self._eqpar.twiss.spos
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
        dic['spos'] = self._eqpar.twiss.spos
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
            spos = self._eqpar.twiss.spos
            acc = spos*0.0 + val
        else:
            raise TypeError('Wrong value for energy acceptance')
        self._accepy = _dcopy(dict(spos=spos, acc=acc))

    @property
    def touschek_data(self):
        """Calculate loss rate due to Touschek beam lifetime.

        parameters used in calculation:

        emit0        = Natural emittance [m.rad]
        energy       = Bunch energy [GeV]
        nr_part      = Number of electrons ber bunch
        espread      = relative energy spread,
        bunlen       = bunch length [m]
        coupling     = emittance coupling factor (emity = coupling*emitx)
        accepen      = relative energy acceptance of the machine.

        twiss = pyaccel.TwissList object or similar object with fields:
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
        en_accep = self.accepen
        twiss = self._eqpar.twiss
        coup = self.coupling
        emit0 = self.emit0
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
        etay = _np.interp(s_calc, twiss.spos[ind], twiss.etay[ind])

        # Volume do bunch
        sigy = _np.sqrt(etay**2*espread**2 + betay*emit0*(coup/(1+coup)))
        sigx = _np.sqrt(etax**2*espread**2 + betax*emit0*(1/(1+coup)))
        vol = bunlen * sigx * sigy

        # Tamanho betatron horizontal do bunch
        sigxb = emit0 * betax / (1+coup)

        fator = betax*etaxl + alphax*etax
        a_var = 1 / (4*espread**2) + (etax**2 + fator**2) / (4*sigxb)
        b_var = betax*fator / (2*sigxb)
        c_var = betax**2 / (4*sigxb) - b_var**2 / (4*a_var)

        # Limite de integração inferior
        ksip = (2*_np.sqrt(c_var)/gamma * d_accp)**2
        ksin = (2*_np.sqrt(c_var)/gamma * d_accn)**2

        # Interpola d_touschek
        d_pos = _np.interp(
            ksip, self._KSI_TABLE, self._D_TABLE, left=0.0, right=0.0)
        d_neg = _np.interp(
            ksin, self._KSI_TABLE, self._D_TABLE, left=0.0, right=0.0)

        # Tempo de vida touschek inverso
        const = (_cst.electron_radius**2 * _cst.light_speed) / (8*_np.pi)
        ratep = const * nr_part/gamma**2 / d_accp**3 * d_pos / vol
        raten = const * nr_part/gamma**2 / d_accn**3 * d_neg / vol
        rate = (ratep + raten) / 2

        # Tempo de vida touschek inverso médio
        avg_rate = _np.trapz(rate, x=s_calc) / (s_calc[-1] - s_calc[0])
        return dict(rate=rate, avg_rate=avg_rate, volume=vol, pos=s_calc)

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
        twiss = self._eqpar.twiss
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
        coupling = emittances ratio
        emit0    = transverse emittance [m·rad]
        taux     = horizontal damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_x = self.accepx
        coupling = self.coupling
        emit0 = self.emit0
        taux = self.taux

        spos = accep_x['spos']
        accep_x = accep_x['acc']

        ksi_x = accep_x / (2*emit0) * (1+coupling)
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
        coupling = emittances ratio
        emit0    = transverse emittance [m·rad]
        tauy     = vertical damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_y = self.accepy
        coupling = self.coupling
        emit0 = self.emit0
        tauy = self.tauy

        spos = accep_y['spos']
        accep_y = accep_y['acc']

        ksi_y = accep_y / (2*emit0) * (1+coupling)/coupling
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
        lft = 1 / self.lossrate_touschek if self.lossrate_touschek > 0 else float('inf')
        return lft

    @property
    def lifetime_elastic(self):
        """Return elastic lifetime [s]."""
        lft = 1 / self.lossrate_elastic if self.lossrate_elastic > 0 else float('inf')
        return lft

    @property
    def lifetime_inelastic(self):
        """Return inelastic lifetime [s]."""
        lft = 1 / self.lossrate_inelastic if self.lossrate_inelastic > 0 else float('inf')
        return lft

    @property
    def lifetime_quantum(self):
        """Return quandtum lifetime [s]."""
        lft = 1 / self.lossrate_quantum if self.lossrate_quantum > 0 else float('inf')
        return lft

    @property
    def lifetime_total(self):
        """Return total lifetime [s]."""
        lft = 1 / self.lossrate_total if self.lossrate_total > 0 else float('inf')
        return lft

    @classmethod
    def get_touschek_integration_table(cls):
        """Return Touschek interpolation table."""
        cls._load_touschek_integration_table()
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
