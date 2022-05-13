"""."""
import time as _time
from functools import partial as _partial

import numpy as _np
import scipy.integrate as _integrate
import scipy.special as _special

from mathphys.functions import get_namedtuple as _get_namedtuple
from mathphys import units as _u, beam_optics as _beam
from mathphys.constants import light_speed as _LSPEED, \
    electron_radius as _ERADIUS

from . import optics as _optics


class IBS:
    """."""

    OPTICS = _get_namedtuple('Optics', ['EdwardsTeng', 'Twiss'])
    EQPARAMS = _get_namedtuple('EqParams', ['BeamEnvelope', 'RadIntegrals'])
    IBS_MODEL = _get_namedtuple('IBSModel', ['CIMP', 'Bane', 'BM'])

    def __init__(
            self, accelerator, curr_per_bunch=0.1e-3,
            ibs_model=IBS_MODEL._fields[0], type_eqparams=EQPARAMS._fields[0],
            type_optics=OPTICS._fields[0]):
        """."""
        self._acc = accelerator
        self._type_eqparams = None
        self._eqparams_func = None
        self._eqpar_data = None
        self._type_optics = None
        self._optics_func = None
        self._optics_data = None
        self._ibs_model = None
        self._ibs_data = None
        self._tau1 = self._tau2 = self._tau3 = None
        self._emit10 = self._emit20 = self._espread0 = self._bunlen0 = None
        self._max_num_iters = 1000
        self._relative_tolerance = 1e-5
        self._delta_time = 1/10

        self._curr_per_bun = curr_per_bunch  # [A]
        self.type_optics = type_optics
        self.type_eqparams = type_eqparams
        self.ibs_model = ibs_model

    def __str__(self):
        """Print all relevant parameters for calculation and results."""
        stg = ''
        stg += '{:30s} {:^20s}\n'.format(
            'type_eqparams:', self.type_eqparams_str)
        stg += '{:30s} {:^20s}\n'.format(
            'type_optics:', self.type_optics_str)
        stg += '{:30s} {:^20s}\n'.format(
            'ibs_model:', self.ibs_model_str)
        stg += '{:30s} {:^20.1f}\n'.format(
            'curr_per_bunch [mA]:', self.curr_per_bunch*1e3)
        stg += 'Radiation damping times considered:\n'
        stg += '    {:30s} {:^20.1f}\n'.format(
            'tau1 (x-plane) [ms]:', self.tau1*1e3)
        stg += '    {:30s} {:^20.1f}\n'.format(
            'tau2 (y-plane) [ms]:', self.tau2*1e3)
        stg += '    {:30s} {:^20.1f}\n'.format(
            'tau3 (l-plane) [ms]:', self.tau3*1e3)
        stg += 'Initial (w/o IBS) equilibrium parameters considered:\n'
        stg += '    {:30s} {:^20.1f}\n'.format(
            'emit10 (x-plane) [pm.rad]:', self.emit10*1e12)
        stg += '    {:30s} {:^20.1f}\n'.format(
            'emit20 (y-plane) [pm.rad]:', self.emit20*1e12)
        stg += '    {:30s} {:^20.4f}\n'.format(
            'espread0 [%]:', self.espread0*1e2)
        stg += '    {:30s} {:^20.3f}\n'.format(
            'bunlen0 [mm]:', self.bunlen0*1e3)
        stg += 'Parameters related to algorithm convergence:\n'
        stg += '    {:30s} {:^20.1f}\n'.format(
            'delta_time (frac. tau123):', self.delta_time)
        stg += '    {:30s} {:^20d}\n'.format(
            'max_num_iters:', self.max_num_iters)
        stg += '    {:30s} {:^20.1g}\n'.format(
            'relative_tolerance:', self.relative_tolerance)
        if self.ibs_data is None:
            stg += 'Final Equilibrium parameters not calculated yet.'
            return stg
        stg += 'Final (with IBS) equilibrium parameters:\n'
        stg += '    {:30s} {:^20.1f}\n'.format(
            'emit1 (x-plane) [pm.rad]:', self.emit1*1e12)
        stg += '    {:30s} {:^20.1f}\n'.format(
            'emit2 (y-plane) [pm.rad]:', self.emit2*1e12)
        stg += '    {:30s} {:^20.4f}\n'.format(
            'espread [%]:', self.espread*1e2)
        stg += '    {:30s} {:^20.3f}\n'.format(
            'bunlen [mm]:', self.bunlen*1e3)
        return stg

    @property
    def relative_tolerance(self):
        """Return the relative tolerance used in checking convergence."""
        return self._relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, value):
        self._relative_tolerance = value

    @property
    def max_num_iters(self):
        """Return the maximum number of iterations of the time evolution."""
        return self._max_num_iters

    @max_num_iters.setter
    def max_num_iters(self, value):
        self._max_num_iters = int(value)

    @property
    def delta_time(self):
        """Delta time used in TimeEvolution, in units of damping times."""
        return self._delta_time

    @delta_time.setter
    def delta_time(self, value):
        self._delta_time = value

    @property
    def type_eqparams_str(self):
        """Which class to use to calculate natural equilibrium parameters.

        The options are `BeamEnvelope' and `RadIntegrals`, which will imply
        the use of `pyaccel.optics.EqParamsFromBeamEnvelope` and
        `pyaccel.optics.EqParamsFromRadIntegrals` respectively.

        It is worth remembering that all equilibrium parameters and damping
        times that will be used in calculations can be overwritten manually
        using the respective properties in this class.

        """
        return IBS.EQPARAMS._fields[self._type_eqparams]

    @property
    def type_eqparams(self):
        """Which class to use to calculate natural equilibrium parameters.

        The options are `BeamEnvelope' and `RadIntegrals`, which will imply
        the use of `pyaccel.optics.EqParamsFromBeamEnvelope` and
        `pyaccel.optics.EqParamsFromRadIntegrals` respectively.

        It is worth remembering that all equilibrium parameters and damping
        times that will be used in calculations can be overwritten manually
        using the respective properties in this class.

        """
        return self._type_eqparams

    @type_eqparams.setter
    def type_eqparams(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._type_eqparams = int(value in IBS.EQPARAMS._fields[1])
        elif int(value) in IBS.EQPARAMS:
            self._type_eqparams = int(value)
        else:
            return
        if self._type_eqparams == self.EQPARAMS.BeamEnvelope:
            self._eqparams_func = _optics.EqParamsFromBeamEnvelope
        elif self._type_eqparams == self.EQPARAMS.RadIntegrals:
            self._eqparams_func = _optics.EqParamsFromRadIntegrals
        self._eqpar_data = self._eqparams_func(self._acc)
        self._ibs_data = None

    @property
    def eqparams_data(self):
        """Equilibrium parameters."""
        return self._eqpar_data

    @property
    def type_optics_str(self):
        """Which class to use to parametrize the 1-turn matrix.

        The options are `EdwardsTeng' and `Twiss`, which will imply
        the use of `pyaccel.optics.calc_edwards_teng` and
        `pyaccel.optics.calc_twiss` respectively.

        """
        return IBS.OPTICS._fields[self._type_optics]

    @property
    def type_optics(self):
        """Which class to use to parametrize the 1-turn matrix.

        The options are `EdwardsTeng' and `Twiss`, which will imply
        the use of `pyaccel.optics.calc_edwards_teng` and
        `pyaccel.optics.calc_twiss` respectively.

        """
        return self._type_optics

    @type_optics.setter
    def type_optics(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._type_optics = int(value in IBS.OPTICS._fields[1])
        elif int(value) in IBS.OPTICS:
            self._type_optics = int(value)
        else:
            return

        if self._type_optics == self.OPTICS.EdwardsTeng:
            self._optics_func = _optics.calc_edwards_teng
        elif self._type_optics == self.OPTICS.Twiss:
            self._optics_func = _optics.calc_twiss
        self._optics_data, *_ = self._optics_func(self._acc, indices='closed')
        self._ibs_data = None

    @property
    def optics_data(self):
        """Optics data. May be Twiss or EdwardsTeng object."""
        return self._optics_data

    @property
    def ibs_model_str(self):
        """Model used to calculate IBS effects. May be CIMP or Bane."""
        return IBS.IBS_MODEL._fields[self._ibs_model]

    @property
    def ibs_model(self):
        """Model used to calculate IBS effects. May be CIMP or Bane."""
        return self._ibs_model

    @ibs_model.setter
    def ibs_model(self, value):
        if value is None:
            return
        if isinstance(value, str) and value in IBS.IBS_MODEL._fields:
            self._ibs_model = IBS.IBS_MODEL._fields.index(value)
        elif int(value) in IBS.IBS_MODEL:
            self._ibs_model = int(value)
        self._ibs_data = None

    @property
    def accelerator(self):
        """Accelerator model."""
        return self._acc

    @accelerator.setter
    def accelerator(self, val):
        self._eqpar_data = self._eqparams_func(val)
        self._optics_data, *_ = self._optics_func(val, indices='closed')
        self._ibs_data = None

    @property
    def curr_per_bunch(self):
        """Return current per bunch used for IBS effects calculation [A]."""
        return self._curr_per_bun

    @curr_per_bunch.setter
    def curr_per_bunch(self, val):
        self._curr_per_bun = float(val)
        self._ibs_data = None

    @property
    def particles_per_bunch(self):
        """Return number of particles per bunch."""
        return int(_beam.calc_number_of_electrons(
            self._acc.energy * _u.eV_2_GeV, self.curr_per_bunch*1e3,
            self._acc.length))

    @property
    def emit10(self):
        """Return initial emittance of mode 1 for calculations [m.rad]."""
        if self._emit10 is not None:
            return self._emit10
        attr = 'emitx' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'emit1'
        return getattr(self._eqpar_data, attr)

    @emit10.setter
    def emit10(self, val):
        self._emit10 = float(val)
        self._ibs_data = None

    @property
    def emit20(self):
        """Return initial emittance of mode 2 for calculations [m.rad]."""
        if self._emit20 is not None:
            return self._emit20
        attr = 'emity' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'emit2'
        return getattr(self._eqpar_data, attr)

    @emit20.setter
    def emit20(self, val):
        self._emit20 = float(val)
        self._ibs_data = None

    @property
    def espread0(self):
        """Return initial relative energy spread for calculations."""
        if self._espread0 is not None:
            return self._espread0
        return self._eqpar_data.espread0

    @espread0.setter
    def espread0(self, val):
        self._espread0 = float(val)
        self._ibs_data = None

    @property
    def bunlen0(self):
        """Return initial bunch length for calculations [m]."""
        if self._bunlen0 is not None:
            return self._bunlen0
        return self._eqpar_data.bunlen

    @bunlen0.setter
    def bunlen0(self, val):
        self._bunlen0 = float(val)
        self._ibs_data = None

    @property
    def tau1(self):
        """Return mode 1 (x) damping time to be used in calculations [s]."""
        if self._tau1 is not None:
            return self._tau1
        attr = 'taux' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'tau1'
        return getattr(self._eqpar_data, attr)

    @tau1.setter
    def tau1(self, val):
        self._tau1 = float(val)
        self._ibs_data = None

    @property
    def tau2(self):
        """Return mode 2 (y) damping time to be used in calculations [s]."""
        if self._tau2 is not None:
            return self._tau2
        attr = 'tauy' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'tau2'
        return getattr(self._eqpar_data, attr)

    @tau2.setter
    def tau2(self, val):
        self._tau2 = float(val)
        self._ibs_data = None

    @property
    def tau3(self):
        """Return mode 3 (long) damping time to be used in calculations [s]."""
        if self._tau3 is not None:
            return self._tau3
        attr = 'taue' if \
            self.type_eqparams == self.EQPARAMS.RadIntegrals else 'tau3'
        return getattr(self._eqpar_data, attr)

    @tau3.setter
    def tau3(self, val):
        self._tau3 = float(val)
        self._ibs_data = None

    @property
    def emit1(self):
        """Emittance of mode 1 (x-plane) with IBS effects included [m.rad]."""
        if self._ibs_data is not None:
            return self._ibs_data['emit1'][-1]

    @property
    def emit2(self):
        """Emittance of mode 2 (y-plane) with IBS effects included [m.rad]."""
        if self._ibs_data is not None:
            return self._ibs_data['emit2'][-1]

    @property
    def espread(self):
        """Energy spread with IBS effects included."""
        if self._ibs_data is not None:
            return self._ibs_data['espread'][-1]

    @property
    def bunlen(self):
        """Longitudinal bunch length with IBS effects included [m]."""
        if self._ibs_data is not None:
            return self._ibs_data['bunlen'][-1]

    @property
    def ibs_data(self):
        """Return dictionary with IBS calculation data."""
        if self._ibs_data is not None:
            return self._ibs_data

    def calc_ibs(self, print_progress=False):
        """Calculate IBS effect on equilibrium parameters.

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
            finalEmit   equilibrium values for Bane and CIMP models[e10 e20
            sigmaE sigmaz]
                    [m rad] [m rad] [] [m]
            relEmit     variation of initial and final values (in %)

        """
        gamma = self._acc.gamma_factor
        beta_factor = self._acc.beta_factor
        alpha = self._eqpar_data.etac
        espread0 = self.espread0
        bunlen0 = self.bunlen0
        e10 = self.emit10
        e20 = self.emit20
        e30 = espread0 * bunlen0  # zero current longitudinal emittance

        # We need to divide damping times by 2 because we are interested in
        # emittance damping times, not on phase space variables damping times.
        rate_sr_1 = 2 / self.tau1
        rate_sr_2 = 2 / self.tau2
        rate_sr_3 = 2 / self.tau3

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
        spos, idx = _np.unique(optics.spos, return_index=True)
        circum = spos[-1] - spos[0]
        beta1 = getattr(optics, names[0])[idx]
        alpha1 = getattr(optics, names[1])[idx]
        eta1 = getattr(optics, names[2])[idx]
        eta1l = getattr(optics, names[3])[idx]
        beta2 = getattr(optics, names[4])[idx]
        alpha2 = getattr(optics, names[5])[idx]
        eta2 = getattr(optics, names[6])[idx]
        eta2l = getattr(optics, names[7])[idx]
        curh_1 = self._calc_curlyh(beta1, alpha1, eta1, eta1l)
        curh_2 = self._calc_curlyh(beta2, alpha2, eta2, eta2l)
        phi1 = eta1l + alpha1*eta1/beta1
        phi2 = eta2l + alpha2*eta2/beta2

        cst = _ERADIUS**2*_LSPEED * num_part
        cst_bane = cst / 16 / gamma**3
        cst_cimp = cst / 64 / _np.pi**2 / gamma**4 / beta_factor**3

        # Total time and time step are set based on assumptions that IBS
        # tipical times are greater (about 10 times) than damping times
        dt_ = _np.min([1/rate_sr_1, 1/rate_sr_2, 1/rate_sr_3])
        dt_ *= self.delta_time

        emits = []
        emits.append([e10, e20, espread0, bunlen0])
        growth_rates = []
        residues = []

        e1_ = e10
        e2_ = e20
        e3_ = e30
        se_ = espread0
        for i in range(self.max_num_iters):
            t0_ = _time.time()
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

            if self._ibs_model == self.IBS_MODEL.Bane:
                amp = cst_bane/(e1_*e2_)**(3/4)/e3_/se_**2

                g_ = self.calc_g_func_bane(alpha, method='exact')

                rate_3 = amp * log * sig_H * g_ * (beta1*beta2)**(-1/4)
                rate_1 = se_**2 * curh_1 / e1_ * rate_3
                rate_2 = se_**2 * curh_2 / e2_ * rate_3
            elif self._ibs_model == self.IBS_MODEL.CIMP:
                amp = cst_cimp / e1_ / e2_ / e3_

                # G function of eq. XX of ref. [1]
                g_ab = self.calc_g_func_cimp(alpha)
                g_ba = self.calc_g_func_cimp(1/alpha)

                factor1 = 2*_np.pi**(3/2) * amp * log
                factor2 = (g_ba/a_ + g_ab/b_) * sig2_H

                rate_1 = factor1*(curh_1/e1_*factor2 - a_*g_ba)
                rate_2 = factor1*(curh_2/e2_*factor2 - b_*g_ab)
                rate_3 = factor1*factor2 / se_**2
            else:
                l1_11 = beta1/e1_ * sig2_H/gamma/gamma
                l1_12 = -phi1*beta1/e1_ * sig2_H / gamma
                l1_22 = curh_1/e1_ * sig2_H

                l2_22 = curh_2/e2_ * sig2_H
                l2_23 = -phi2*beta2/e2_ * sig2_H / gamma
                l2_33 = beta2/e2_ * sig2_H/gamma/gamma

                l3_22 = 1/se_/se_ * sig2_H

                args = _np.array([
                    l1_11, l1_12, l1_22, l2_22, l2_23, l2_33, l3_22]).T
                rate_1, rate_2, rate_3 = self._calc_bm_integral(args)

                amp = cst_cimp / e1_ / e2_ / e3_
                fac = 4*_np.pi * amp * log
                rate_1 *= fac
                rate_2 *= fac
                rate_3 *= fac

            rate_1_avg = _np.trapz(rate_1, x=spos) / circum
            rate_2_avg = _np.trapz(rate_2, x=spos) / circum
            rate_3_avg = _np.trapz(rate_3, x=spos) / circum

            exc_e1 = e1_ - e10
            exc_e2 = e2_ - e20
            exc_e3 = e3_ - e30

            dlt_e1 = (2*e1_*rate_1_avg - exc_e1*rate_sr_1) * dt_
            dlt_e2 = (2*e2_*rate_2_avg - exc_e2*rate_sr_2) * dt_
            dlt_e3 = (2*e3_*rate_3_avg - exc_e3*rate_sr_3) * dt_

            residue = _np.abs([dlt_e1/e1_, dlt_e2/e2_, dlt_e3/e3_])
            e1_ += dlt_e1
            e2_ += dlt_e2
            e3_ += dlt_e3

            se_ = _np.sqrt(e3_ / conv_sige2sigs)
            bl_ = conv_sige2sigs * se_
            emits.append([e1_, e2_, se_, bl_])
            growth_rates.append([rate_1, rate_2, rate_3])
            residues.append(residue)

            if print_progress:
                print(
                    f'iter: {i:04d}, '
                    f'relative change [%]: {residue.max()*100:10.5f}'
                    f', ET: {_time.time()-t0_:15.3f}s')

            if residue.max() < self.relative_tolerance:
                break

        emits = _np.array(emits)
        residues = _np.array(residues)
        growth_rates = _np.array(growth_rates)

        tim = _np.arange(len(emits)) * dt_
        self._ibs_data = dict(
            tim=tim, emit1=emits[:, 0], emit2=emits[:, 1], espread=emits[:, 2],
            bunlen=emits[:, 3], residues=residues, growth_rates=growth_rates,
            spos=spos)
        return self._ibs_data

    @classmethod
    def _calc_bm_integral(cls, args):
        vec, _ = _integrate.quad_vec(
            _partial(cls._get_bm_integrand, args=args),
            0, _np.inf, workers=1, norm='max')
        siz = args.shape[0]
        rate1 = vec[:siz]
        rate2 = vec[siz:2*siz]
        rate3 = vec[2*siz:]
        return rate1, rate2, rate3

    @staticmethod
    def _get_bm_integrand(lamb, args=None):
        # import sympy as sp
        # a, b, c, d, e = sp.symbols('a b c d e')
        # m = sp.Matrix([[a, d, 0], [d, b, e], [0, e, c]])
        # detm = sp.det(m)
        # print(detm)
        # sp.simplify(m.inv() * detm)
        l1_11, l1_12, l1_22, l2_22, l2_23, l2_33, l3_22 = args.T
        lt_11 = l1_11 + lamb
        lt_22 = 1 + lamb
        lt_33 = l2_33 + lamb
        lt_12 = l1_12
        lt_23 = l2_23

        det_t = lt_11 * lt_22 * lt_33
        det_t -= lt_11 * lt_23 * lt_23
        det_t -= lt_33 * lt_12 * lt_12

        im11 = lt_33*lt_22 - lt_23*lt_23
        im22 = lt_11*lt_33
        im33 = lt_11*lt_22 - lt_12*lt_12
        im12 = -lt_33*lt_12
        # im13 = lt_12*lt_23  # not needed here
        im23 = -lt_11*lt_23

        tr_im = im11 + im22 + im33

        prefac = (lamb / det_t)**0.5 / det_t

        # Remember that: Tr(AB) = Tr(BA) = a_ij*b_ji
        integ3 = prefac * (l3_22*tr_im - 3*l3_22*im22)
        integ2 = prefac * (
            (l2_22 + l2_33)*tr_im -
            3*(l2_22*im22 + l2_33*im33 + 2*l2_23*im23))
        integ1 = prefac * (
            (l1_11 + l1_22)*tr_im -
            3*(l1_11*im11 + l1_22*im22 + 2*l1_12*im12))
        return _np.array([integ1, integ2, integ3]).ravel()

    @classmethod
    def calc_g_func_cimp(cls, omega):
        """_summary_

        Args:
            omega (_type_): _description_

        Returns:
            _type_: _description_

        """
        arg = (omega*omega + 1) / (2*omega)
        # We have to use the type 3 legendre functions because the argument
        # arg above is always larger than 1, which means it is along the
        # branch cut of type 2 legendre functions.
        p_0 = cls._calc_associated_legendre_function(
            arg, mu=0, nu=-1/2, type_=3)
        p_m1 = cls._calc_associated_legendre_function(
            arg, mu=-1, nu=-1/2, type_=3)

        idx = omega >= 1
        ret = p_0
        ret[idx] += 3/2 * p_m1[idx]
        ret[~idx] -= 3/2 * p_m1[~idx]
        ret *= _np.sqrt(_np.pi/omega)
        return ret

    @staticmethod
    def calc_g_func_bane(alphas, method='exact'):
        """_summary_

        Args:
            alphas (_type_): _description_
            method (str, optional): _description_. Defaults to 'exact'.

        Raises:
            ValueError: when method not in {`exact`, `fitting`, `numeric`}.

        Returns:
            _type_: _description_

        """
        ret = _np.zeros(alphas.size)
        if method.lower().startswith('num'):
            lim = 1000
            for i, alpha in enumerate(alphas):
                ret[i], _ = _integrate.quad(
                    lambda u: 1/_np.sqrt(1+u*u)/_np.sqrt(alpha*alpha + u*u),
                    0, _np.inf, limit=lim, )
            ret *= 2 * _np.sqrt(alphas)/_np.pi
            return ret

        # Bane points out in ref. [2] that g(alpha) = g(1/alpha). We will use
        # this fact for both calculation methods below:
        alphas = alphas.copy()
        idx = alphas > 1
        alphas[idx] = 1/alphas[idx]

        if method.lower().startswith('exa'):
            return 2*_np.sqrt(alphas)/_np.pi * _special.ellipk(1-alphas*alphas)
        elif method.lower().startswith('fit'):
            return alphas**(0.021 - 0.044*_np.log(alphas))
        else:
            raise ValueError(
                "Wrong value for method. Must be in {'exact', "
                "'fitting', 'numeric'}")

    @staticmethod
    def _calc_associated_legendre_function(z, mu=0, nu=-1/2, type_=3):
        r"""Calculate the associated Legendre function P_nu^mu.

        The variable `mu` is the order and the variable `nu` is the degree of
        the associated Legendre function.

        Implementation is based on the general version of Legendre functions
        in terms of the hypergeometric function 2F1, given by reference [1].

        To see an explanation on the interpretation of the variable `type_`,
        please refer to the discussion in section Details in reference [2].
        It is related to how branch cuts are chosen for the analytic
        continuation of the Legendre function outside the unit circle, given
        that it has regular poles at -1 and 1:
         - In `type_ == 2` there is two branch cuts: in the intervals
            [-infinity, -1] U [1, infinity]
         - `type_ == 3`, means one branch cut in [-infinity, 1]

        Reference [3] also provides explicit formulas for `type_ == 3`
        functions and for x inside the unit circle.

        References:
            [1] Wikipedia contributors, "Legendre function," Wikipedia, The
                Free Encyclopedia,
                https://en.wikipedia.org/w/index.php?title=Legendre_function&oldid=1060272134
                (accessed May 2, 2022).
            [2] Wolfram Alpha,
                https://reference.wolfram.com/language/ref/LegendreP.html.
            [3] F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I.
                Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V.
                Saunders, H. S. Cohl, and M. A. McClain, editors, "NIST
                Digital Library of Mathematical Functions". Release 1.1.5 of
                2022-03-15. Available at http://dlmf.nist.gov/, Chapter 14.3.

        Args:
            z (numpy.ndarray, complex): Complex valued argument.
            mu (complex, optional): Order (lower symbol). Defaults to 0.
            nu (complex \ -N, optional): Degree (upper symbol).
                Defaults to -1/2.
            type_ (int, optional): Type of the associated Legendre
                function. May assume values in {2, 3}. Defaults to 3.

        Raises:
            ValueError: when `type_` is not in {2, 3}.

        Returns:
            numpy.ndarray: the associated legendre function P_nu^mu(z).

        """
        ret = _special.hyp2f1(-nu, nu + 1, 1-mu, (1 - z)/2)
        if _np.allclose(_np.abs(mu), 0):
            return ret

        ret /= _special.gamma(1-mu)
        if type_ in {1, 2}:
            ret *= ((1 + z)/(1 - z))**(mu/2)
        elif type_ == 3:
            idx = _np.abs(z) < 1
            ret[idx] *= ((1 + z[idx])/(1 - z[idx]))**(mu/2)
            ret[~idx] *= ((z[~idx] + 1)/(z[~idx] - 1))**(mu/2)
        else:
            raise ValueError('Incorrect value for variable `type_`.')
        return ret

    @staticmethod
    def _calc_curlyh(beta, alpha, eta, etal, spos=None):
        gamma = (1 + alpha*alpha) / beta
        curlyh = gamma*eta*eta + 2*alpha*eta*etal + beta*etal*etal
        if spos is None:
            return curlyh
        avg_h = _np.trapz(curlyh, x=spos) / (spos[-1] - spos[0])
        return curlyh, avg_h
