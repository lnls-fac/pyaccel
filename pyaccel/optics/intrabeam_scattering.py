"""."""

import os as _os
import importlib as _implib
from copy import deepcopy as _dcopy

import numpy as _np
import scipy.integrate as _integrate
import scipy.special as _special

from mathphys.functions import get_namedtuple as _get_namedtuple
from mathphys import units as _u, beam_optics as _beam
from mathphys.constants import light_speed as _LSPEED, \
    electron_radius as _ERADIUS

from . import optics as _optics


class IBS:

    _D_TOUSCHEK_FILE = _os.path.join(
        _os.path.dirname(__file__), 'data', 'd_touschek.npz')

    _KSI_TABLE = None
    _D_TABLE = None

    OPTICS = _get_namedtuple('Optics', ['EdwardsTeng', 'Twiss'])
    EQPARAMS = _get_namedtuple('EqParams', ['BeamEnvelope', 'RadIntegrals'])
    IBS_MODEL = _get_namedtuple('IBSModel', ['CIMP', 'Bane'])

    def __init__(self, accelerator, ibs_model=None,
                 type_eqparams=None, type_optics=None):
        """."""
        self._acc = accelerator

        self._type_eqparams = IBS.EQPARAMS.BeamEnvelope
        self._type_optics = IBS.OPTICS.EdwardsTeng
        self._ibs_model = IBS.IBS_MODEL.CIMP
        self.type_eqparams = type_eqparams
        self.type_optics = type_optics
        self.ibs_model = ibs_model

        if self.type_eqparams == self.EQPARAMS.BeamEnvelope:
            self._eqparams_func = _optics.EqParamsFromBeamEnvelope
        elif self.type_eqparams == self.EQPARAMS.RadIntegrals:
            self._eqparams_func = _optics.EqParamsFromRadIntegrals

        if self.type_optics == self.OPTICS.EdwardsTeng:
            self._optics_func = _optics.calc_edwards_teng
        elif self._type_optics == self.OPTICS.Twiss:
            self._optics_func = _optics.calc_twiss

        self._eqpar = self._eqparams_func(self._acc)
        self._optics_data, *_ = self._optics_func(self._acc, indices='closed')
        self._curr_per_bun = 100/864  # [mA]
        self._tau1 = self._tau2 = self._tau3 = None
        self._emit1 = self._emit2 = self._espread0 = self._bunlen = None

    @property
    def type_eqparams_str(self):
        """."""
        return IBS.EQPARAMS._fields[self._type_eqparams]

    @property
    def type_eqparams(self):
        """."""
        return self._type_eqparams

    @type_eqparams.setter
    def type_eqparams(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._type_eqparams = int(value in IBS.EQPARAMS._fields[1])
        elif int(value) in IBS.EQPARAMS:
            self._type_eqparams = int(value)

    @property
    def type_optics_str(self):
        """."""
        return IBS.OPTICS._fields[self._type_optics]

    @property
    def type_optics(self):
        """."""
        return self._type_optics

    @type_optics.setter
    def type_optics(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._type_optics = int(value in IBS.OPTICS._fields[1])
        elif int(value) in IBS.OPTICS:
            self._type_optics = int(value)

    @property
    def ibs_model_str(self):
        """."""
        return IBS.IBS_MODEL._fields[self._ibs_model]

    @property
    def ibs_model(self):
        """."""
        return self._ibs_model

    @ibs_model.setter
    def ibs_model(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._ibs_model = int(
                value in IBS.IBS_MODEL._fields[1])
        elif int(value) in IBS.IBS_MODEL:
            self._ibs_model = int(value)

    @property
    def accelerator(self):
        """."""
        return self._acc

    @accelerator.setter
    def accelerator(self, val):
        self._eqpar = self._eqparams_func(val)
        self._optics_data, *_ = self._optics_func(val, indices='closed')

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

    @staticmethod
    def _calc_curlyh(beta, alpha, eta, etal, spos=None):
        gamma = (1 + alpha*alpha) / beta
        curlyh = gamma*eta*eta + 2*alpha*eta*etal + beta*etal*etal
        if spos is None:
            return curlyh
        avg_h = _np.trapz(curlyh, x=spos) / (spos[-1] - spos[0])
        return curlyh, avg_h

    def calc_ibs(self, ratio=0.5):
        """Calculate the horizontal, vertical emittance and energy spread growth.

        The code uses two approximated models: CIMP and high energy developed
        by Bane. Equilibrium emittances are calculated by the temporal
        evolution over a time which is a multiple of damping time, based on
        equations described in KIM - A Code for Calculating the Time Evolution
        of Beam Parameters in High Intensity Circular Accelerators - PAC 97.
        The bunch length was calculated by the expression given in SANDS - The
        Physics of Electron Storage Rings, an Introduction.

        This function was based very much on lnls_calcula_ibs developed by
        Afonso H. C. Mukai which calculates the effect of IBS with CIMP model
        only.

        INPUT
            data_atsum      struct with ring parameters (atsummary):
                            revTime              revolution time [s]
                            gamma
                            twiss
                            compactionFactor
                            damping
                            naturalEnergySpread
                            naturalEmittance     zero current emittance [m rad]
                            radiationDamping     damping times [s]
                            synctune
            I           Beam Current [mA]
            K           Coupling      [%]
            R(=1)       growth bunch length factor (optional)

        OUTPUT
            finalEmit   equilibrium values for Bane and CIMP models[e10 e20 sigmaE sigmaz]
                    [m rad] [m rad] [] [m]
            relEmit     variation of initial and final values (in %)

        """
        # Load interpolation tables
        # Function g(alpha) of Bane's model (elliptic integral)
        self._load_table_bane()
        x_table_bane = self._X_TABLE_BANE
        g_table_bane = self._G_TABLE_BANE

        # Function g of CIMP model (Related to the associated Legendre
        # functions)
        self._load_table_cimp()
        x_table_cimp = self._X_TABLE_CIMP
        g_table_cimp = self._G_TABLE_CIMP

        self._load_touschek_integration_table()

        # Take parameters of database
        gamma = self._acc.gamma_factor
        beta_factor = self._acc.beta_factor
        alpha = self.eqparams.etac
        espread0 = self.espread0
        bunlen0 = self.bunlen
        e10 = self.emit1
        e20 = self.emit2
        e20_coup = e20 * ratio
        e20_disp = e20 * (1-ratio)
        coup = e20_coup / e10
        e30 = espread0 * bunlen0  # zero current longitudinal emittance
        tau_1 = self.tau1
        tau_2 = self.tau2
        tau_3 = self.tau3
        num_part = self.particles_per_bunch
        optics = self._optics_data
        # Conversion factor from energy spread to bunch length, assuming a
        # linear voltage. It will be used along the code to convert from
        # longitudinal emittance to energy spread.
        # This term is equal to w_s/eta_c/c
        conv_sige2sigs = bunlen0 / espread0

        twi_names = [
            'betax', 'alphax', 'etax', 'etapx',
            'betay', 'alphay', 'etay', 'etapy']
        edteng_names = [
            'beta1', 'alpha1', 'eta1', 'etap1',
            'beta2', 'alpha2', 'eta2', 'etap2']
        names = twi_names if \
            self.type_optics == self.OPTICS.Twiss else edteng_names

        # Copy values making them unique
        spos, idx = _np.unique(self.eqparams.twiss.spos, return_index=True)
        beta1 = getattr(optics, names[0])[idx]
        alpha1 = getattr(optics, names[1])[idx]
        eta1 = getattr(optics, names[2])[idx]
        eta1l = getattr(optics, names[3])[idx]
        beta2 = getattr(optics, names[4])[idx]
        alpha2 = getattr(optics, names[5])[idx]
        eta2 = getattr(optics, names[6])[idx]
        eta2l = getattr(optics, names[7])[idx]

        circum = spos[-1] - spos[0]

        cst = _ERADIUS**2*_LSPEED * num_part
        cst_bane = cst / 16 / gamma**3
        cst_cimp = cst / 64 / _np.pi**2 / gamma**4 / beta_factor**3

        # Total time and time step are set based on assumptions that IBS
        # tipical times are greater (about 10 times) than damping times
        ibs_ov_rad = 10  # ratio of IBS times over damping times
        tf = ibs_ov_rad * _np.max([tau_1, tau_2, tau_3])
        dt_ = _np.min([tau_1, tau_2, tau_3]) / ibs_ov_rad
        niter = _np.ceil(tf / dt_)
        t = _np.linspace(0, tf, niter)

        emits = _np.zeros((4, niter), dtype=float)
        emits[:, 0] = [e10, e20, espread0, bunlen0]

        e1_ = e10
        e2_ = e20
        e3_ = e30
        se_ = espread0
        for i in range(1, niter):
            curh_1, avgh_1 = self._calc_curlyh(
                beta1, alpha1, eta1, eta1l, spos=spos)
            curh_2, avgh_2 = self._calc_curlyh(
                beta2, alpha2, eta2, eta2l, spos=spos)

            sig2_H = 1 / (curh_1/e1_ + curh_2/e2_ + 1/se_**2)
            sig_H = _np.sqrt(sig2_H)

            a_ = sig_H * _np.sqrt(beta1/e1_) / gamma
            b_ = sig_H * _np.sqrt(beta2/e2_) / gamma
            # Argument of g(alpha) functions (for Bane and CIMP)
            alpha = a_ / b_

            # According to discussion after eq. 22 in ref. [1] the parameter d
            # is taken to be the vertical beamsize.
            d = _np.sqrt(e2_*beta2 + (se_*eta2)**2)  # == sigy
            q_sqr = sig2_H * beta_factor**2 * 2*d/_ERADIUS
            log = _np.log(q_sqr / a_**2)

            if self._ibs_model == self.IBS_MODEL.BANE:
                amp = cst_bane/(e1_*e2_)**(3/4)/e3_/se_**2

                g_ = _np.interp(alpha, x_table_bane, g_table_bane)

                rate_3 = amp * log * sig_H * g_ * (beta1*beta2)**(-1/4)
                rate_3 = _np.trapz(rate_3, x=spos) / circum
                time_3 = 1 / rate_3  # longitudinal growth time Bane

                rate_1 = se_**2 * avgh_1 / e1_ * rate_3
                time_1 = 1 / rate_1  # horizontal growth time Bane

                rate_2 = se_**2 * avgh_2 / e2_ * rate_3
                time_2 = 1 / rate_2  # Vertical growth time Bane
            else:
                amp = cst_cimp / e1_ / e2_ / e3_

                g_ab = _np.interp(alpha, x_table_cimp, g_table_cimp)
                g_ba = _np.interp(1/alpha, x_table_cimp, g_table_cimp)

                factor1 = 2*_np.pi**(3/2) * amp * log
                factor2 = (g_ba/a_ + g_ab/b_) * sig2_H

                rate_3 = factor1 * factor2 / se_**2
                rate_3 = _np.trapz(rate_3, x=spos) / circum
                time_3 = 1 / rate_3  # longitudinal growth time CIMP

                rate_1 = factor1*(curh_1/e1_*factor2 - a_*g_ba)
                rate_1 = _np.trapz(rate_1, x=spos) / circum
                time_1 = 1 / rate_1  # horizontal groth time CIMP

                rate_2 = (curh_2/e2_*factor2 - b_*g_ab)
                rate_2 *= factor1
                rate_2 = _np.trapz(rate_2, x=spos) / circum
                time_2 = 1 / rate_2  # horizontal groth time CIMP

            exc_e3 = e3_ - e30
            exc_e1 = e1_ - e10
            exc_e2 = e2_ - e20_disp - e1_*coup/(1+coup)*ratio

            dlt_e3 = e3_/time_3 - exc_e3/tau_3
            dlt_e1 = e1_/time_1 - exc_e1/tau_1
            dlt_e2 = e2_/time_2 - exc_e2/tau_2

            e3_ += dlt_e3 * dt_
            e1_ += dlt_e1 * dt_
            e2_ += dlt_e2 * dt_
            se_ = _np.sqrt(e3_ / conv_sige2sigs)
            bl_ = conv_sige2sigs * se_
            emits[:, i] = [e1_, e2_, se_, bl_]

            if not (i % 50):
                print('.\n')
            else:
                print('.')
        return emits

    @classmethod
    def get_touschek_integration_table(cls, ksi_ini=None, ksi_end=None):
        """Return Touschek interpolation table."""
        if None in (ksi_ini, ksi_end):
            cls._load_touschek_integration_table()
        else:
            cls._calc_d_touschek_table(ksi_ini, ksi_end)
        return cls._KSI_TABLE, cls._D_TABLE

    @classmethod
    def calc_g_func_cimp(cls, omega):
        arg = (omega*omega + 1) / (2*omega)
        ret = _np.sqrt(_np.pi/omega)* (_special.lpmv(0, -0.5, arg))

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

    def _plot_ibs(self):
        EBANE(:,1) = EBANE(:,1) * 1e12;
        EBANE(:,2) = EBANE(:,2) * 1e12;
        EBANE(:,3) = EBANE(:,3) * 10000;
        ECIMP(:,1) = ECIMP(:,1) * 1e12;
        ECIMP(:,2) = ECIMP(:,2) * 1e12;
        ECIMP(:,3) = ECIMP(:,3) * 10000;

        t_min = t(1);
        t_max = t(end);
        f_min = 0.96;
        f_max = 1.04;

        figure;
        plot(t,EBANE(:,1),'color',[1.0 0.0 0.0]);
        title('Horizontal Emittance');
        xlabel('t [s]');
        ylabel('\epsilon_x [pm rad]');
        axis([t_min t_max f_min*min(EBANE(:,1)) f_max*max(EBANE(:,1))]);

        hold on

        plot(t,ECIMP(:,1),'color',[0.0 0.0 1.0]);

        legend('BANE','CIMP','location','east');
        hold off

        figure;
        plot(t,EBANE(:,2),'color',[1.0 0.0 0.0]);
        title('Vertical Emittance');
        xlabel('t [s]');
        ylabel('\epsilon_y [pm rad]');
        axis([t_min t_max f_min*min(EBANE(:,2)) f_max*max(EBANE(:,2))]);

        hold on

        plot(t,ECIMP(:,2),'color',[0.0 0.0 1.0]);
        legend('BANE','CIMP','location','east');

        hold off

        figure;
        plot(t,EBANE(:,3),'color',[1.0 0.0 0.0]);
        title('Energy Spread');
        xlabel('t [s]');
        ylabel('\sigma_E / E_0 [10^{-4}]');
        axis([t_min t_max f_min*min(EBANE(:,3)) f_max*max(EBANE(:,3))]);

        hold on

        plot(t,ECIMP(:,3),'color',[0.0 0.0 1.0]);
        legend('BANE','CIMP','location','east');

        hold off
