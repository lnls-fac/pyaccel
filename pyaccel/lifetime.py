"""Beam lifetime calculation."""

import os as _os
import importlib as _implib
from copy import deepcopy as _dcopy
import numpy as _np
from mathphys.functions import get_namedtuple as _get_namedtuple

from mathphys import constants as _cst, units as _u, \
    beam_optics as _beam

from . import optics as _optics

if _implib.util.find_spec('scipy'):
    import scipy.integrate as _integrate
    import scipy.special as _special
else:
    _integrate = None
    _special = None


class Lifetime:
    """Class which calculates the lifetime for a given accelerator."""

    # Constant factors
    _MBAR_2_PASCAL = 1.0e-3 / _u.pascal_2_bar

    _D_TOUSCHEK_FILE = _os.path.join(
        _os.path.dirname(__file__), 'data', 'd_touschek.npz')

    _KSI_TABLE = None
    _D_TABLE = None

    OPTICS = _get_namedtuple('Optics', ['EdwardsTeng', 'Twiss'])
    EQPARAMS = _get_namedtuple('EqParams', ['BeamEnvelope', 'RadIntegrals'])
    TOUSCHEKMODEL = _get_namedtuple('TouschekModel', ['Piwinski', 'FlatBeam'])

    DEFAULT_BEAM_CURRENT = 100  # [mA]
    DEFAULT_AVG_PRESSURE = 1e-9  # [mbar]
    DEFAULT_ATOMIC_NUMBER = 7  # Nitrogen
    DEFAULT_TEMPERATURE = 300  # [K]

    def __init__(self, accelerator, touschek_model=None,
                 eqparams=None, type_optics=None):
        """."""
        self._acc = accelerator

        self._type_eqparams = None
        self._eqparams_func = None
        self._eqpar = None
        self._process_eqparams(eqparams)

        self._type_optics = Lifetime.OPTICS.EdwardsTeng
        self._touschek_model = Lifetime.TOUSCHEKMODEL.Piwinski
        self.type_optics = type_optics
        self.touschek_model = touschek_model

        if self.type_optics == self.OPTICS.EdwardsTeng:
            self._optics_func = _optics.calc_edwards_teng
        elif self._type_optics == self.OPTICS.Twiss:
            self._optics_func = _optics.calc_twiss
        self._optics_data, *_ = self._optics_func(self._acc, indices='closed')
        _twiss = self._optics_data
        if self.type_optics != self.OPTICS.Twiss:
            _twiss, *_ = _optics.calc_twiss(self._acc, indices='closed')
        res = _optics.calc_transverse_acceptance(self._acc, _twiss)
        self._accepx_nom = _np.min(res[0])
        self._accepy_nom = _np.min(res[1])
        self._curr_per_bun = self.DEFAULT_BEAM_CURRENT / self._acc.harmonic_number
        self._avg_pressure = self.DEFAULT_AVG_PRESSURE
        self._atomic_number = self.DEFAULT_ATOMIC_NUMBER
        self._temperature = self.DEFAULT_TEMPERATURE
        self._tau1 = self._tau2 = self._tau3 = None
        self._emit1 = self._emit2 = self._espread0 = self._bunlen = None
        self._accepx = self._accepy = self._accepen = None

    @property
    def type_eqparams_str(self):
        """."""
        return Lifetime.EQPARAMS._fields[self._type_eqparams]

    @property
    def type_eqparams(self):
        """."""
        return self._type_eqparams

    @type_eqparams.setter
    def type_eqparams(self, value):
        self._process_eqparams(value)

    @property
    def type_optics_str(self):
        """."""
        return Lifetime.OPTICS._fields[self._type_optics]

    @property
    def type_optics(self):
        """."""
        return self._type_optics

    @type_optics.setter
    def type_optics(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._type_optics = int(value in Lifetime.OPTICS._fields[1])
        elif int(value) in Lifetime.OPTICS:
            self._type_optics = int(value)

    @property
    def touschek_model_str(self):
        """."""
        return Lifetime.TOUSCHEKMODEL._fields[self._touschek_model]

    @property
    def touschek_model(self):
        """."""
        return self._touschek_model

    @touschek_model.setter
    def touschek_model(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._touschek_model = int(
                value in Lifetime.TOUSCHEKMODEL._fields[1])
        elif int(value) in Lifetime.TOUSCHEKMODEL:
            self._touschek_model = int(value)

    @property
    def accelerator(self):
        """."""
        return self._acc

    @accelerator.setter
    def accelerator(self, val):
        if self._eqparams_func is not None:
            self._eqpar = self._eqparams_func(val)
        self._optics_data, *_ = self._optics_func(val, indices='closed')
        _twiss = self._optics_data
        if self.type_optics != self.OPTICS.Twiss:
            _twiss, *_ = _optics.calc_twiss(val, indices='closed')
        res = _optics.calc_transverse_acceptance(val, _twiss)
        self._accepx_nom = _np.min(res[0])
        self._accepy_nom = _np.min(res[1])
        self._acc = val

    @property
    def equi_params(self):
        """Equilibrium parameters."""
        return self._eqpar

    @property
    def optics_data(self):
        """Optics data."""
        return self._optics_data

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
    def emit1(self):
        """Stationary emittance of mode 1 [m.rad]."""
        if self._emit1 is not None:
            return self._emit1
        attr = 'emitx' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'emit1'
        return getattr(self._eqpar, attr)

    @emit1.setter
    def emit1(self, val):
        self._emit1 = float(val)

    @property
    def emit2(self):
        """Stationary emittance of mode 2 [m.rad]."""
        if self._emit2 is not None:
            return self._emit2
        attr = 'emity' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'emit2'
        return getattr(self._eqpar, attr)

    @emit2.setter
    def emit2(self, val):
        self._emit2 = float(val)

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
    def tau1(self):
        """Mode 1 damping Time [s]."""
        if self._tau1 is not None:
            return self._tau1
        attr = 'taux' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'tau1'
        return getattr(self._eqpar, attr)

    @tau1.setter
    def tau1(self, val):
        self._tau1 = float(val)

    @property
    def tau2(self):
        """Mode 2 damping Time [s]."""
        if self._tau2 is not None:
            return self._tau2
        attr = 'tauy' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'tau2'
        return getattr(self._eqpar, attr)

    @tau2.setter
    def tau2(self, val):
        self._tau2 = float(val)

    @property
    def tau3(self):
        """Mode 3 damping Time [s]."""
        if self._tau3 is not None:
            return self._tau3
        attr = 'taue' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'tau3'
        return getattr(self._eqpar, attr)

    @tau3.setter
    def tau3(self, val):
        self._tau3 = float(val)

    @property
    def accepen(self):
        """Longitudinal acceptance."""
        if self._accepen is not None:
            return self._accepen
        dic = dict()
        rf_accep = self._eqpar.rf_acceptance
        dic['spos'] = self._optics_data.spos
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
            spos = self._optics_data.spos
            accp = spos*0.0 + val[1]
            accn = spos*0.0 + val[0]
        elif isinstance(val, (int, _np.int_, float, _np.float_)):
            spos = self._optics_data.spos
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
        dic['spos'] = self._optics_data.spos
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
        elif isinstance(val, (int, _np.int_, float, _np.float_)):
            spos = self._optics_data.spos
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
        dic['spos'] = self._optics_data.spos
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
        elif isinstance(val, (int, _np.int_, float, _np.float_)):
            spos = self._optics_data.spos
            acc = spos*0.0 + val
        else:
            raise TypeError('Wrong value for energy acceptance')
        self._accepy = _dcopy(dict(spos=spos, acc=acc))

    @property
    def touschek_data(self):
        """Calculate loss rate due to Touschek beam lifetime.

        If touschek_model = 'FlatBeam', the calculation follows the formulas
            presented in Ref. [1], where the vertical betatron beam size and
            vertical dispersion are not taken into account
        If touschek_model = 'Piwinski', the calculation follows the formulas
            presented in Ref. [2], Eqs. 32-42. This formalism describes the
            most general case with respect to the horizontal and vertical
            betatron oscillation, the horizontal and vertical dispersion, and
            the derivatives of the amplitude functions and dispersions.

        References:
            [1] Duff, J. Le. (1988). Single and multiple Touschek effects. In
                CERN Acccelerator School: Accelerator Physics (pp. 114–130).
            [2] Piwinski, A. (1999). The Touschek Effect in Strong Focusing
                Storage Rings. November. http://arxiv.org/abs/physics/9903034

        parameters used in calculation:

        emit1        = Mode 1 emittance [m.rad]
        emit2        = Mode 2 emittance [m.rad]
        energy       = Bunch energy [GeV]
        nr_part      = Number of electrons ber bunch
        espread      = relative energy spread,
        bunlen       = bunch length [m]
        accepen      = relative energy acceptance of the machine.

        optics = pyaccel.TwissArray object or similar object with fields:
                spos, betax, betay, etax, etay, alphax, alphay, etapx, etapy

                or

                pyaccel.EdwardsTengArray object or similar object with fields:
                spos, beta1, beta2, eta1, eta2, alpha1, alpha2, etap1, etap2

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
            volume   = volume of the beam along the ring [m^3]
            touschek_coeffs = dict with coefficients for corresponding
                formalism
        """
        self._load_touschek_integration_table()
        gamma = self._acc.gamma_factor
        beta = self._acc.beta_factor
        en_accep = self.accepen
        optics = self._optics_data
        emit1, emit2 = self.emit1, self.emit2
        espread = self.espread0
        bunlen = self.bunlen
        nr_part = self.particles_per_bunch

        _, ind = _np.unique(optics.spos, return_index=True)
        spos = en_accep['spos']
        accp = en_accep['accp']
        accn = en_accep['accn']

        # calculate lifetime for each 10cm of the ring
        npoints = int((spos[-1] - spos[0])/0.1)
        s_calc = _np.linspace(spos[0], spos[-1], npoints)
        d_accp = _np.interp(s_calc, spos, accp)
        d_accn = _np.interp(s_calc, spos, -accn)

        # if momentum aperture is 0, set it to 1e-4:
        d_accp[d_accp == 0] = 1e-4
        d_accn[d_accn == 0] = 1e-4

        twi_names = [
            'betax', 'alphax', 'etax', 'etapx',
            'betay', 'alphay', 'etay', 'etapy']
        edteng_names = [
            'beta1', 'alpha1', 'eta1', 'etap1',
            'beta2', 'alpha2', 'eta2', 'etap2']
        names = twi_names if \
            self.type_optics == self.OPTICS.Twiss else edteng_names

        s_ind = optics.spos[ind]
        beta1 = _np.interp(s_calc, s_ind, getattr(optics, names[0])[ind])
        alpha1 = _np.interp(s_calc, s_ind, getattr(optics, names[1])[ind])
        eta1 = _np.interp(s_calc, s_ind, getattr(optics, names[2])[ind])
        eta1l = _np.interp(s_calc, s_ind, getattr(optics, names[3])[ind])

        beta2 = _np.interp(s_calc, s_ind, getattr(optics, names[4])[ind])
        alpha2 = _np.interp(s_calc, s_ind, getattr(optics, names[5])[ind])
        eta2 = _np.interp(s_calc, s_ind, getattr(optics, names[6])[ind])
        eta2l = _np.interp(s_calc, s_ind, getattr(optics, names[7])[ind])

        # Betatron bunch sizes
        sig1b2 = emit1 * beta1
        sig2b2 = emit2 * beta2

        # Bunch volume
        sig2 = _np.sqrt(eta2**2*espread**2 + beta2*emit2)
        sig1 = _np.sqrt(eta1**2*espread**2 + beta1*emit1)
        vol = bunlen * sig1 * sig2
        const = (_cst.electron_radius**2 * _cst.light_speed) / (8*_np.pi)

        touschek_coeffs = dict()
        if self.touschek_model == self.TOUSCHEKMODEL.FlatBeam:
            fator = beta1*eta1l + alpha1*eta1
            a_var = 1 / (4*espread**2) + (eta1**2 + fator**2) / (4*sig1b2)
            b_var = beta1*fator / (2*sig1b2)
            c_var = beta1**2 / (4*sig1b2) - b_var**2 / (4*a_var)

            # Lower integration limit
            ksip = (2*_np.sqrt(c_var)/gamma * d_accp)**2
            ksin = (2*_np.sqrt(c_var)/gamma * d_accn)**2

            # Interpolate d_touschek
            d_pos = _np.interp(
                ksip, self._KSI_TABLE, self._D_TABLE, left=0.0, right=0.0)
            d_neg = _np.interp(
                ksin, self._KSI_TABLE, self._D_TABLE, left=0.0, right=0.0)

            touschek_coeffs['a_var'] = a_var
            touschek_coeffs['b_var'] = b_var
            touschek_coeffs['c_var'] = c_var
            touschek_coeffs['ksip'] = ksip
            touschek_coeffs['ksin'] = ksin
            touschek_coeffs['d_pos'] = d_pos
            touschek_coeffs['d_neg'] = d_neg

            # Touschek rate
            ratep = const * nr_part/gamma**2 / d_accp**3 * d_pos / vol
            raten = const * nr_part/gamma**2 / d_accn**3 * d_neg / vol
            rate = (ratep+raten)/2
        elif self.touschek_model == self.TOUSCHEKMODEL.Piwinski:
            eta1til2 = (alpha1*eta1 + beta1*eta1l)**2
            eta2til2 = (alpha2*eta2 + beta2*eta2l)**2
            espread2 = espread*espread
            betagamma2 = (beta*gamma)**2

            val1 = 1/espread2
            val2 = (eta1*eta1 + eta1til2)/(sig1b2)
            val3 = (eta2*eta2 + eta2til2)/(sig2b2)
            sigh2 = 1/(val1 + val2 + val3)

            c1_ = beta1**2/sig1b2*(1-sigh2*eta1til2/sig1b2)
            c2_ = beta2**2/sig2b2*(1-sigh2*eta2til2/sig2b2)
            ch_ = (sig1*sig2)**2 - (espread2*eta1*eta2)**2

            b1_ = (c1_ + c2_)/(2*betagamma2)
            b2_ = (c1_ - c2_)**2/4
            b2_ += eta1til2*eta2til2*(sigh2/emit1/emit2)**2
            b2_ /= (betagamma2**2)
            b2_ = _np.sqrt(b2_)

            taum_p = (beta*d_accp)**2
            taum_n = (beta*d_accn)**2

            f_int_p = self.f_integral_simps(taum_p, b1_, b2_)
            f_int_n = self.f_integral_simps(taum_n, b1_, b2_)

            touschek_coeffs['b1'] = b1_
            touschek_coeffs['b2'] = b2_
            touschek_coeffs['taum_p'] = taum_p
            touschek_coeffs['taum_n'] = taum_n
            touschek_coeffs['f_int_p'] = f_int_p
            touschek_coeffs['f_int_n'] = f_int_n

            rate_const = const * nr_part/gamma**2/bunlen
            rate_const /= _np.sqrt(ch_)
            ratep = rate_const * f_int_p/taum_p
            raten = rate_const * f_int_n/taum_n
            rate = (ratep + raten)/2

        rate = _np.array(rate).ravel()
        # Average inverse Touschek Lifetime
        avg_rate = _np.trapz(rate, x=s_calc) / (s_calc[-1] - s_calc[0])
        dit = dict(
            rate=rate, avg_rate=avg_rate,
            volume=vol, pos=s_calc,
            touschek_coeffs=touschek_coeffs)
        return dit

    @staticmethod
    def f_function_arg(kappa, kappam, b1_, b2_):
        """Integrand in the F(taum, B1, B2) expression.

        Argument of the integral of F(taum, B1, B2) function of Eq. (42) from
        Ref. [2] of touschek_data property documentation.

        In order to improve the numerical integration speed, the
        transformation tau = tan(kappa)^2 and taum = tan(kappam)^2 is made
        resulting in the expression right below Eq. (42) in Ref. [2] (equation
        without number). This argument is integrated from kappam to pi/2 in
        the method f_integral_simps of this class.
        """
        tau = (_np.tan(kappa)**2)[:, None]
        taum = (_np.tan(kappam)**2)[None, :]
        ratio = tau/taum/(1+tau)
        arg = (2*tau+1)**2 * (ratio - 1)/tau
        arg += tau - _np.sqrt(tau*taum*(1+tau))
        arg -= (2+1/(2*tau))*_np.log(ratio)
        arg *= _np.sqrt(1+tau)
        bessel = _np.exp(-(b1_-b2_)*tau)*_special.i0e(b2_*tau)
        return arg * bessel

    @staticmethod
    def f_integral_simps(taum, b1_, b2_):
        """F(taum, B1, B2) function.

        The expression used for F can be found right below Eq. (42) from Ref.
        [2] of touschek_data property documentation. The numerical integration
        from kappam to pi/2 is performed with the Simpson's 3/8 Rule.
        """
        kappam = _np.arctan(_np.sqrt(taum))
        npts = int(3*100)
        dkappa = (_np.pi/2-kappam)/npts
        kappa = _np.linspace(kappam, _np.pi/2, npts+1)
        func = Lifetime.f_function_arg(kappa, kappam, b1_, b2_)

        # Simpson's 3/8 Rule - N must be mod(N, 3) = 0
        val1 = func[0:-1:3, :] + func[3::3, :]
        val2 = func[1::3, :] + func[2::3, :]
        f_int = 3*dkappa/8*_np.sum(val1 + 3*val2, axis=0)

        # # Simpson's 1/3 Rule - N must be mod(N, 2) = 0
        # val1 = func[0::2, :] + func[2::2, :]
        # val2 = func[1::2, :]
        # f_int = dkappa/3*_np.sum(val1+4*val2, axis=0)
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
        optics         = Linear optics parameters

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_x = self.accepx
        accep_y = self.accepy
        pressure = self.avg_pressure
        optics = self._optics_data
        energy = self._acc.energy
        beta = self._acc.beta_factor
        atomic_number = self.atomic_number
        temperature = self.temperature

        if self.type_optics == self.OPTICS.Twiss:
            beta1, beta2 = optics.betax, optics.betay
        else:
            beta1, beta2 = optics.beta1, optics.beta2
        energy_joule = energy / _u.joule_2_eV

        spos = optics.spos
        _, idx = _np.unique(accep_x['spos'], return_index=True)
        _, idy = _np.unique(accep_y['spos'], return_index=True)
        accep_x = _np.interp(spos, accep_x['spos'][idx], accep_x['acc'][idx])
        accep_y = _np.interp(spos, accep_y['spos'][idy], accep_y['acc'][idy])

        thetax = _np.sqrt(accep_x/beta1)
        thetay = _np.sqrt(accep_y/beta2)
        ratio = thetay / thetax

        f_x = 2*_np.arctan(ratio) + _np.sin(2*_np.arctan(ratio))
        f_x *= pressure * self._MBAR_2_PASCAL * beta1 / accep_x
        f_y = _np.pi - 2*_np.arctan(ratio) + _np.sin(2*_np.arctan(ratio))
        f_y *= pressure * self._MBAR_2_PASCAL * beta2 / accep_y

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
        Based on Ref[1].

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

        References:
            [1] Beam losses and lifetime - A. Piwinski, em CERN 85-19.
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
        emit1    = mode 1 emittance [m·rad]
        tau1     = mode 1 damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_x = self.accepx
        emit1 = self.emit1
        tau1 = self.tau1

        spos = accep_x['spos']
        accep_x = accep_x['acc']

        ksi_x = accep_x / (2*emit1)
        rate = self._calc_quantum_loss_rate(ksi_x, tau1)

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
        emit2    = mode 2 emittance [m·rad]
        tau2     = mode 2 damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        accep_y = self.accepy
        emit2 = self.emit2
        tau2 = self.tau2

        spos = accep_y['spos']
        accep_y = accep_y['acc']

        ksi_y = accep_y / (2*emit2)
        rate = self._calc_quantum_loss_rate(ksi_y, tau2)

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
        tau3      = mode 3 damping time [s]

        output:

        dictionary with fields:
            rate     = loss rate along the ring [1/s]
            avg_rate = average loss rate along the ring [1/s]
            pos      = longitudinal position where loss rate was calculated [m]
        """
        en_accep = self.accepen
        espread = self.espread0
        tau3 = self.tau3

        spos = en_accep['spos']
        accp = en_accep['accp']
        accn = en_accep['accn']

        ratep = self._calc_quantum_loss_rate((accp/espread)**2 / 2, tau3)
        raten = self._calc_quantum_loss_rate((accn/espread)**2 / 2, tau3)
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

    def _process_eqparams(self, eqparams):
        # equilibrium parameters argument
        beamenv_set = (_optics.EqParamsFromBeamEnvelope, _optics.EqParamsNormalModes)
        radiint_set = (_optics.EqParamsFromRadIntegrals, _optics.EqParamsXYModes)
        if eqparams in (None, Lifetime.EQPARAMS.BeamEnvelope):
            self._type_eqparams = Lifetime.EQPARAMS.BeamEnvelope
            self._eqparams_func = _optics.EqParamsFromBeamEnvelope
            self._eqpar = self._eqparams_func(self._acc)
        elif isinstance(eqparams, beamenv_set):
            self._type_eqparams = Lifetime.EQPARAMS.BeamEnvelope
            self._eqparams_func = None
            self._eqpar = eqparams
        elif eqparams == Lifetime.EQPARAMS.RadIntegrals:
            self._type_eqparams = eqparams
            self._eqparams_func = _optics.EqParamsFromRadIntegrals
            self._eqpar = self._eqparams_func(self._acc)
        elif isinstance(eqparams, radiint_set):
            self._type_eqparams = Lifetime.EQPARAMS.RadIntegrals
            self._eqparams_func = None
            self._eqpar = eqparams
        else:
            raise ValueError('Invalid equilibrium parameter argument!')

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
