"""Optics module."""

import sys as _sys
import math as _math
import numpy as _np

import mathphys as _mp
import trackcpp as _trackcpp

from . import lattice as _lattice
from . import tracking as _tracking
from . import accelerator as _accelerator
from .utils import interactive as _interactive


class OpticsException(Exception):
    """."""


class Twiss:
    """."""

    def __init__(self, twiss=None, copy=True):
        """."""
        if twiss is None:
            self._t = _trackcpp.Twiss()
        elif isinstance(twiss, Twiss):
            self._t = twiss._t
        elif isinstance(twiss, _trackcpp.Twiss):
            self._t = twiss
        else:
            raise TypeError(
                'twiss must be a trackcpp.Twiss or a Twiss object.')
        if twiss is not None and copy:
            self._t = _trackcpp.Twiss(self._t)

    @property
    def spos(self):
        """Return spos."""
        return self._t.spos

    @spos.setter
    def spos(self, value):
        """Set spos."""
        self._t.spos = value

    @property
    def rx(self):
        """."""
        return self._t.co.rx

    @rx.setter
    def rx(self, value):
        """."""
        self._t.co.rx = value

    @property
    def ry(self):
        """."""
        return self._t.co.ry

    @ry.setter
    def ry(self, value):
        """."""
        self._t.co.ry = value

    @property
    def px(self):
        """."""
        return self._t.co.px

    @px.setter
    def px(self, value):
        """."""
        self._t.co.px = value

    @property
    def py(self):
        """."""
        return self._t.co.py

    @py.setter
    def py(self, value):
        """."""
        self._t.co.py = value

    @property
    def de(self):
        """."""
        return self._t.co.de

    @de.setter
    def de(self, value):
        """."""
        self._t.co.de = value

    @property
    def dl(self):
        """."""
        return self._t.co.dl

    @dl.setter
    def dl(self, value):
        """."""
        self._t.co.dl = value

    @property
    def co(self):
        """."""
        return _np.array([
            self.rx, self.px, self.ry, self.py, self.de, self.dl])

    @co.setter
    def co(self, value):
        """."""
        self.rx, self.px = value[0], value[1]
        self.ry, self.py = value[2], value[3]
        self.de, self.dl = value[4], value[5]

    @property
    def betax(self):
        """."""
        return self._t.betax

    @betax.setter
    def betax(self, value):
        """."""
        self._t.betax = value

    @property
    def betay(self):
        """."""
        return self._t.betay

    @betay.setter
    def betay(self, value):
        """."""
        self._t.betay = value

    @property
    def alphax(self):
        """."""
        return self._t.alphax

    @alphax.setter
    def alphax(self, value):
        """."""
        self._t.alphax = value

    @property
    def alphay(self):
        """."""
        return self._t.alphay

    @alphay.setter
    def alphay(self, value):
        """."""
        self._t.alphay = value

    @property
    def mux(self):
        """."""
        return self._t.mux

    @mux.setter
    def mux(self, value):
        """."""
        self._t.mux = value

    @property
    def muy(self):
        """."""
        return self._t.muy

    @muy.setter
    def muy(self, value):
        """."""
        self._t.muy = value

    @property
    def etax(self):
        """."""
        return self._t.etax[0]

    @etax.setter
    def etax(self, value):
        """."""
        self._t.etax[0] = value

    @property
    def etay(self):
        """."""
        return self._t.etay[0]

    @etay.setter
    def etay(self, value):
        """."""
        self._t.etay[0] = value

    @property
    def etapx(self):
        """."""
        return self._t.etax[1]

    @etapx.setter
    def etapx(self, value):
        """."""
        self._t.etax[1] = value

    @property
    def etapy(self):
        """."""
        return self._t.etay[1]

    @etapy.setter
    def etapy(self, value):
        """."""
        self._t.etay[1] = value

    def make_dict(self):
        """."""
        cod = self.co
        beta = [self.betax, self.betay]
        alpha = [self.alphax, self.alphay]
        etax = [self.etax, self.etapx]
        etay = [self.etay, self.etapy]
        mus = [self.mux, self.muy]
        return {
            'co': cod, 'beta': beta, 'alpha': alpha,
            'etax': etax, 'etay': etay, 'mu': mus}

    @staticmethod
    def make_new(*args, **kwrgs):
        """Build a Twiss object."""
        if args:
            if isinstance(args[0], dict):
                kwrgs = args[0]
        twi = Twiss()
        twi.co = kwrgs.get('co', (0.0,)*6)
        twi.mux, twi.muy = kwrgs.get('mu', (0.0, 0.0))
        twi.betax, twi.betay = kwrgs.get('beta', (0.0, 0.0))
        twi.alphax, twi.alphay = kwrgs.get('alpha', (0.0, 0.0))
        twi.etax, twi.etapx = kwrgs.get('etax', (0.0, 0.0))
        twi.etay, twi.etapy = kwrgs.get('etay', (0.0, 0.0))
        return twi

    def __str__(self):
        """."""
        rst = ''
        rst += 'spos          : ' + '{0:+10.3e}'.format(self.spos)
        fmt = '{0:+10.3e}, {1:+10.3e}'
        rst += '\nrx, ry        : ' + fmt.format(self.rx, self.ry)
        rst += '\npx, py        : ' + fmt.format(self.px, self.py)
        rst += '\nde, dl        : ' + fmt.format(self.de, self.dl)
        rst += '\nmux, muy      : ' + fmt.format(self.mux, self.muy)
        rst += '\nbetax, betay  : ' + fmt.format(self.betax, self.betay)
        rst += '\nalphax, alphay: ' + fmt.format(self.alphax, self.alphay)
        rst += '\netax, etapx   : ' + fmt.format(self.etax, self.etapx)
        rst += '\netay, etapy   : ' + fmt.format(self.etay, self.etapy)
        return rst

    def __eq__(self, other):
        """."""
        if not isinstance(other, Twiss):
            return NotImplemented
        for attr in self._t.__swig_getmethods__:
            self_attr = getattr(self, attr)
            if isinstance(self_attr, _np.ndarray):
                if (self_attr != getattr(other, attr)).any():
                    return False
            else:
                if self_attr != getattr(other, attr):
                    return False
        return True


class TwissList:
    """."""

    def __init__(self, twiss_list=None):
        """Read-only list of matrices.

        Keyword argument:
        twiss_list -- trackcpp Twiss vector (default: None)
        """
        # TEST!
        if twiss_list is None:
            self._tl = _trackcpp.CppTwissVector()
        if isinstance(twiss_list, _trackcpp.CppTwissVector):
            self._tl = twiss_list
        else:
            raise OpticsException('invalid Twiss vector')
        self._ptl = [self._tl[i] for i in range(len(self._tl))]

    def __len__(self):
        """."""
        return len(self._tl)

    def __getitem__(self, index):
        """."""
        if isinstance(index, (int, _np.int_)):
            return Twiss(twiss=self._tl[index], copy=False)
        elif isinstance(index, (list, tuple, _np.ndarray)) and \
                all(isinstance(x, (int, _np.int_)) for x in index):
            tl = _trackcpp.CppTwissVector()
            for i in index:
                tl.append(self._tl[i])
            return TwissList(twiss_list=tl)
        elif isinstance(index, slice):
            return TwissList(twiss_list=self._tl[index])
        else:
            raise TypeError('invalid index')

    def append(self, value):
        """."""
        if isinstance(value, _trackcpp.Twiss):
            self._tl.append(value)
            self._ptl.append(value)
        elif isinstance(value, Twiss):
            self._tl.append(value._t)
            self._ptl.append(value._t)
        elif self._is_list_of_lists(value):
            t = _trackcpp.Twiss()
            for line in value:
                t.append(line)
            self._tl.append(t)
            self._ptl.append(t)
        else:
            raise OpticsException('can only append twiss-like objects')

    def _is_list_of_lists(self, value):
        valid_types = (list, tuple)
        if not isinstance(value, valid_types):
            return False
        for line in value:
            if not isinstance(line, valid_types):
                return False
        return True

    @property
    def spos(self):
        """."""
        spos = _np.array([
            float(self._ptl[i].spos) for i in range(len(self._ptl))])
        return spos if len(spos) > 1 else spos[0]

    @property
    def betax(self):
        """."""
        betax = _np.array([
            float(self._ptl[i].betax) for i in range(len(self._ptl))])
        return betax if len(betax) > 1 else betax[0]

    @property
    def betay(self):
        """."""
        betay = _np.array([
            float(self._ptl[i].betay) for i in range(len(self._ptl))])
        return betay if len(betay) > 1 else betay[0]

    @property
    def alphax(self):
        """."""
        alphax = _np.array([
            float(self._ptl[i].alphax) for i in range(len(self._ptl))])
        return alphax if len(alphax) > 1 else alphax[0]

    @property
    def alphay(self):
        """."""
        alphay = _np.array([
            float(self._ptl[i].alphay) for i in range(len(self._ptl))])
        return alphay if len(alphay) > 1 else alphay[0]

    @property
    def mux(self):
        """."""
        mux = _np.array([
            float(self._ptl[i].mux) for i in range(len(self._ptl))])
        return mux if len(mux) > 1 else mux[0]

    @property
    def muy(self):
        """."""
        muy = _np.array([
            float(self._ptl[i].muy) for i in range(len(self._ptl))])
        return muy if len(muy) > 1 else muy[0]

    @property
    def etax(self):
        """."""
        etax = _np.array([
            float(self._ptl[i].etax[0]) for i in range(len(self._ptl))])
        return etax if len(etax) > 1 else etax[0]

    @property
    def etay(self):
        """."""
        etay = _np.array([
            float(self._ptl[i].etay[0]) for i in range(len(self._ptl))])
        return etay if len(etay) > 1 else etay[0]

    @property
    def etapx(self):
        etapx = _np.array([
            float(self._ptl[i].etax[1]) for i in range(len(self._ptl))])
        return etapx if len(etapx) > 1 else etapx[0]

    @property
    def etapy(self):
        etapy = _np.array([
            float(self._ptl[i].etay[1]) for i in range(len(self._ptl))])
        return etapy if len(etapy) > 1 else etapy[0]

    @property
    def rx(self):
        res = _np.array([float(ptl.co.rx) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def ry(self):
        res = _np.array([float(ptl.co.ry) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def px(self):
        res = _np.array([float(ptl.co.px) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def py(self):
        res = _np.array([float(ptl.co.py) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def de(self):
        res = _np.array([float(ptl.co.de) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def dl(self):
        res = _np.array([float(ptl.co.dl) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def co(self):
        co = [self._ptl[i].co for i in range(len(self._ptl))]
        co = [[co[i].rx, co[i].px, co[i].ry, co[i].py, co[i].de, co[i].dl]
              for i in range(len(co))]
        co = _np.transpose(_np.array(co))
        return co if len(co[0, :]) > 1 else co[:, 0]


class EquilibriumParameters:

    def __init__(self, accelerator):
        self._acc = _accelerator.Accelerator()
        self._m66 = None
        self._twi = None
        self._alpha = 0.0
        self._integrals = _np.zeros(6)
        self._damping = _np.zeros(3)
        self._radiation_damping = _np.zeros(3)
        self._bendidx = EquilibriumParameters.get_bend_idx(
            acc=accelerator)
        self.accelerator = _lattice.refine_lattice(
            accelerator=accelerator,
            max_length=0.05, indices=self._bendidx)

    def __str__(self):
        rst = ''
        fmt = '{:<30s}: '
        fmtn = '{:.4g}'

        fmte = fmt + fmtn
        rst += fmte.format('\nEnergy [GeV]', self.accelerator.energy*1e-9)

        ints = 'I1,I2,I3,I3a,I4,I5,I6'.split(',')
        rst += '\n' + fmt.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'Jx,Jy,Je'.split(',')
        rst += '\n' + fmt.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'taux,tauy,taue'.split(',')
        rst += '\n' + fmt.format(', '.join(ints) + ' [ms]')
        rst += ', '.join([fmtn.format(1000*getattr(self, x)) for x in ints])

        rst += fmte.format('\nmomentum compaction x 1e4', self.alpha*1e4)
        rst += fmte.format('\nenergy loss [keV]', self.U0/1000)
        rst += fmte.format('\novervoltage', self.overvoltage)
        rst += fmte.format('\nsync phase [°]', self.syncphase*180/_math.pi)
        rst += fmte.format('\nsync tune', self.synctune)
        rst += fmte.format('\nnatural emittance [nm.rad]', self.emit0*1e9)
        rst += fmte.format('\nnatural espread [%]', self.espread0*100)
        rst += fmte.format('\nbunch length [mm]', self.bunch_length*1000)
        rst += fmte.format('\nRF energy accep. [%]', self.rf_acceptance*100)
        return rst

    @property
    def accelerator(self):
        return self._acc

    @accelerator.setter
    def accelerator(self, acc):
        if isinstance(acc, _accelerator.Accelerator):
            self._acc = acc
            self._calc_radiation_integrals()

    @property
    def twiss(self):
        return self._twi

    @property
    def m66(self):
        return self._m66

    @property
    def I1(self):
        return self._integrals[0]

    @property
    def I2(self):
        return self._integrals[1]

    @property
    def I3(self):
        return self._integrals[2]

    @property
    def I3a(self):
        return self._integrals[3]

    @property
    def I4(self):
        return self._integrals[4]

    @property
    def I5(self):
        return self._integrals[5]

    @property
    def I6(self):
        return self._integrals[6]

    @property
    def Jx(self):
        return 1.0 - self.I4/self.I2

    @property
    def Jy(self):
        return 1.0

    @property
    def Je(self):
        return 2.0 + self.I4/self.I2

    @property
    def alphax(self):
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jx / leng

    @property
    def alphay(self):
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jy / leng

    @property
    def alphae(self):
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Je / leng

    @property
    def taux(self):
        return 1/self.alphax

    @property
    def tauy(self):
        return 1/self.alphay

    @property
    def taue(self):
        return 1/self.alphae

    @property
    def espread0(self):
        Cq = _mp.constants.Cq
        gamma = self._acc.gamma_factor
        return _math.sqrt(Cq * gamma**2 * self.I3 / (2*self.I2 + self.I4))

    @property
    def emit0(self):
        Cq = _mp.constants.Cq
        gamma = self._acc.gamma_factor
        return Cq * gamma**2 * self.I5 / (self.Jx*self.I2)

    @property
    def U0(self):
        E0 = self._acc.energy / 1e9  # in GeV
        rad_cgamma = _mp.constants.rad_cgamma
        return rad_cgamma/(2*_math.pi) * E0**4 * self.I2 * 1e9  # in eV

    @property
    def overvoltage(self):
        v_cav = get_rf_voltage(self._acc)
        return v_cav/self.U0

    @property
    def syncphase(self):
        return _math.pi - _math.asin(1/self.overvoltage)

    @property
    def alpha(self):
        return self._alpha

    @property
    def etac(self):
        gamma = self._acc.gamma_factor
        return 1/(gamma*gamma) - self.alpha

    @property
    def synctune(self):
        E0 = self._acc.energy
        v_cav = get_rf_voltage(self._acc)
        harmon = self._acc.harmonic_number
        return _math.sqrt(
            self.etac*harmon*v_cav*_math.cos(self.syncphase)/(2*_math.pi*E0))

    @property
    def bunch_length(self):
        c = _mp.constants.light_speed
        beta = self._acc.beta_factor
        rev_freq = get_revolution_frequency(self._acc)

        bunlen = beta * c * abs(self.etac) * self.espread0
        bunlen /= 2*_math.pi * self.synctune * rev_freq
        return bunlen

    @property
    def rf_acceptance(self):
        E0 = self._acc.energy
        sph = self.syncphase
        V = get_rf_voltage(self._acc)
        ov = self.overvoltage
        h = self._acc.harmonic_number
        etac = self.etac

        eaccep2 = V * _math.sin(sph) / (_math.pi*h*abs(etac)*E0)
        eaccep2 *= 2 * (_math.sqrt(ov**2 - 1.0) - _math.acos(1.0/ov))
        return _math.sqrt(eaccep2)

    @staticmethod
    def calcH(beta, alpha, x, xl):
        gamma = (1 + alpha**2) / beta
        return beta*xl**2 + 2*alpha*x*xl + gamma*x**2

    @staticmethod
    def get_bend_idx(acc):
        n = len(acc)
        angle = _np.zeros(n)
        for i in range(n):
            angle[i] = acc[i].angle
        idx, *_ = _np.nonzero(angle)
        return idx

    def as_dict(self):
        pars = {
            'twiss',
            'I1', 'I2', 'I3', 'I3a', 'I4', 'I5', 'I6',
            'Jx', 'Jy', 'Je',
            'alphax', 'alphay', 'alphae',
            'taux', 'tauy', 'taue',
            'espread0', 'emit0', 'bunch_length',
            'U0', 'overvoltage', 'syncphase', 'synctune',
            'alpha', 'etac', 'rf_acceptance',
            }
        dic = {par: getattr(self, par) for par in pars}
        dic['energy'] = self.accelerator.energy
        return dic

    def _calc_radiation_integrals(self):
        """Calculate radiation integrals for periodic systems"""

        acc = self._acc
        twi, m66 = calc_twiss(acc, indices='closed')
        self._twi = twi
        self._m66 = m66
        self._alpha = get_mcf(acc)

        spos = _lattice.find_spos(acc, indices='closed')
        etax, etapx, betax, alphax = twi.etax, twi.etapx, twi.betax, twi.alphax

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

        H_in = self.calcH(betax_in, alphax_in, etax_in, etapx_in)
        H_out = self.calcH(betax_out, alphax_out, etax_in, etapx_out)

        etax_avg = (etax_in + etax_out) / 2
        H_avg = (H_in + H_out) / 2
        rho2, rho3 = rho**2, rho**3
        rho3abs = _np.abs(rho3)

        integrals = _np.zeros(7)
        integrals[0] = _np.dot(etax_avg/rho, leng)
        integrals[1] = _np.dot(1/rho2, leng)
        integrals[2] = _np.dot(1/rho3abs, leng)
        integrals[3] = _np.dot(1/rho3, leng)

        integrals[4] = _np.dot(etax_avg/rho3 * (1+2*rho2*K), leng)
        # for general wedge magnets:
        integrals[4] += sum((etax_in/rho2) * _np.tan(angle_in))
        integrals[4] += sum((etax_out/rho2) * _np.tan(angle_out))

        integrals[5] = _np.dot(H_avg / rho3abs, leng)
        integrals[6] = _np.dot((K*etax_avg)**2, leng)

        self._integrals = integrals


@_interactive
def calc_twiss(accelerator=None, init_twiss=None, fixed_point=None,
               indices='open', energy_offset=None):
    """Return Twiss parameters of uncoupled dynamics.

    Keyword arguments:
    accelerator   -- Accelerator object
    init_twiss    -- Twiss parameters at the start of first element
    fixed_point   -- 6D position at the start of first element
    indices       -- Open or closed
    energy_offset -- float denoting the energy deviation (used only for
                    periodic solutions).

    Returns:
    tw -- list of Twiss objects (closed orbit data is in the objects vector)
    m66 -- one-turn transfer matrix

    """
    if indices == 'open':
        closed_flag = False
    elif indices == 'closed':
        closed_flag = True
    else:
        raise OpticsException("invalid value for 'indices' in calc_twiss")

    _m66 = _trackcpp.Matrix()
    _twiss = _trackcpp.CppTwissVector()

    if init_twiss is not None:
        # as a transport line: uses init_twiss
        _init_twiss = init_twiss._t
        if fixed_point is None:
            _fixed_point = _init_twiss.co
        else:
            raise OpticsException(
                'arguments init_twiss and fixed_point are mutually exclusive')
    else:
        # as a periodic system: try to find periodic solution
        if accelerator.harmonic_number == 0:
            raise OpticsException(
                'Either harmonic number was not set or calc_twiss was'
                'invoked for transport line without initial twiss')

        if fixed_point is None:
            _closed_orbit = _trackcpp.CppDoublePosVector()
            _fixed_point_guess = _trackcpp.CppDoublePos()
            if energy_offset is not None:
                _fixed_point_guess.de = energy_offset

            if not accelerator.cavity_on and not accelerator.radiation_on:
                r = _trackcpp.track_findorbit4(
                    accelerator.trackcpp_acc, _closed_orbit,
                    _fixed_point_guess)
            elif not accelerator.cavity_on and accelerator.radiation_on:
                raise OpticsException(
                    'The radiation is on but the cavity is off')
            else:
                r = _trackcpp.track_findorbit6(
                    accelerator.trackcpp_acc, _closed_orbit,
                    _fixed_point_guess)

            if r > 0:
                raise _tracking.TrackingException(
                    _trackcpp.string_error_messages[r])
            _fixed_point = _closed_orbit[0]

        else:
            _fixed_point = _tracking._Numpy2CppDoublePos(fixed_point)
            if energy_offset is not None:
                _fixed_point.de = energy_offset

        _init_twiss = _trackcpp.Twiss()

    r = _trackcpp.calc_twiss(
        accelerator.trackcpp_acc, _fixed_point, _m66, _twiss,
        _init_twiss)
    if r > 0:
        raise OpticsException(_trackcpp.string_error_messages[r])

    if not closed_flag:
        _twiss.pop_back()

    twiss = TwissList(_twiss)
    m66 = _tracking._CppMatrix2Numpy(_m66)

    return twiss, m66


@_interactive
def calc_emittance_coupling(accelerator,
                            mode='fitting',
                            x0=1e-5, y0=1e-8,
                            nr_turns=100):
    # Code copied from:
    # http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    # In order to check the nomenclature used, please go to:
    # https://mathworld.wolfram.com/Ellipse.html
    def fitEllipse(x_dat, y_dat):
        x_dat = x_dat[:, _np.newaxis]
        y_dat = y_dat[:, _np.newaxis]
        mat = _np.hstack((
            x_dat*x_dat, x_dat*y_dat, y_dat*y_dat, x_dat, y_dat,
            _np.ones_like(x_dat)))
        sq_mat = _np.dot(mat.T, mat)
        cof = _np.zeros([6, 6])
        cof[0, 2] = cof[2, 0] = 2
        cof[1, 1] = -1
        eig, eiv = _np.linalg.eig(_np.linalg.solve(sq_mat, cof))
        siz = _np.argmax(_np.abs(eig))
        eiv = eiv[:, siz]
        a, b, c, d, f, g = eiv[0], eiv[1]/2, eiv[2], eiv[3]/2, eiv[4]/2, eiv[5]
        numerator = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        denominator1 = (b*b-a*c) * ((c-a) * _np.sqrt(
            1 + 4*b*b / ((a-c)*(a-c)))-(c+a))
        denominator2 = (b*b-a*c) * ((a-c) * _np.sqrt(
            1 + 4*b*b / ((a-c)*(a-c)))-(c+a))
        axis1 = _np.sqrt(numerator/denominator1)
        axis2 = _np.sqrt(numerator/denominator2)
        return _np.array([axis1, axis2])  # returns the axes of the ellipse

    def calc_emittances(orbit, twiss, idx=0):
        alphax = twiss.alphax[idx]
        betax = twiss.betax[idx]
        gammax = (1 + alphax*alphax)/betax

        alphay = twiss.alphay[idx]
        betay = twiss.betay[idx]
        gammay = (1 + alphay*alphay)/betay

        rxx, pxx, ryy, pyy = orbit[0, :], orbit[1, :], orbit[2, :], orbit[3, :]

        emitx = gammax * rxx**2 + 2 * alphax * rxx * pxx + betax * pxx**2
        emity = gammay * ryy**2 + 2 * alphay * ryy * pyy + betay * pyy**2
        return _np.mean(emitx), _np.mean(emity)

    acc = accelerator[:]
    acc.cavity_on = False
    acc.radiation_on = False

    orb = _tracking.find_orbit4(acc)
    rin = _np.array(
        [x0+orb[0], 0+orb[1], y0+orb[2], 0+orb[3], 0, 0], dtype=float)
    rout, *_ = _tracking.ring_pass(
        acc, rin, nr_turns=nr_turns, turn_by_turn='closed', element_offset=0)

    if mode == 'fitting':
        minx, majx = fitEllipse(rout[0], rout[1])
        miny, majy = fitEllipse(rout[2], rout[3])
        emitx = minx * majx
        emity = miny * majy
    elif mode == 'twiss':
        twiss, *_ = calc_twiss(acc)
        dtraj = rout - _np.vstack((orb, _np.array([[0], [0]])))
        emitx, emity = calc_emittances(dtraj, twiss)
    else:
        raise Exception('Invalid mode, set fitting or twiss')
    return emity / emitx


@_interactive
def get_rf_frequency(accelerator):
    """Return the frequency of the first RF cavity in the lattice"""
    for e in accelerator:
        if e.frequency != 0.0:
            return e.frequency
    raise OpticsException('no cavity element in the lattice')


@_interactive
def get_rf_voltage(accelerator):
    """Return the voltage of the first RF cavity in the lattice"""
    voltages = []
    for e in accelerator:
        if e.voltage != 0.0:
            voltages.append(e.voltage)
    if voltages:
        if len(voltages) == 1:
            return voltages[0]
        else:
            return voltages
    else:
        raise OpticsException('no cavity element in the lattice')


@_interactive
def get_revolution_period(accelerator):
    return accelerator.length/accelerator.velocity


@_interactive
def get_revolution_frequency(accelerator):
    return 1.0/get_revolution_period(accelerator)


@_interactive
def get_traces(accelerator=None,
               m1turn=None, closed_orbit=None, dim='6D',
               energy_offset=0.0):
    """Return traces of 6D one-turn transfer matrix"""
    if m1turn is None:
        if dim == '4D':
            m1turn = _tracking.find_m44(
                accelerator, indices='m44', energy_offset=energy_offset,
                closed_orbit=closed_orbit)
        elif dim == '6D':
            m1turn = _tracking.find_m66(
                accelerator, indices='m66', closed_orbit=closed_orbit)
        else:
            raise Exception('Set valid dimension: 4D or 6D')
    trace_x = m1turn[0, 0] + m1turn[1, 1]
    trace_y = m1turn[2, 2] + m1turn[3, 3]
    trace_z = 2
    if dim == '6D':
        trace_z = m1turn[4, 4] + m1turn[5, 5]
    return trace_x, trace_y, trace_z, m1turn, closed_orbit


@_interactive
def get_frac_tunes(accelerator=None, m1turn=None, dim='6D', closed_orbit=None,
                   energy_offset=0.0, coupled=False):
    """Return fractional tunes of the accelerator"""

    trace_x, trace_y, trace_z, m1turn, closed_orbit = get_traces(
        accelerator, m1turn=m1turn, dim=dim, closed_orbit=closed_orbit,
        energy_offset=energy_offset)
    tune_x = _math.acos(trace_x/2.0)/2.0/_math.pi
    tune_y = _math.acos(trace_y/2.0)/2.0/_math.pi
    tune_z = _math.acos(trace_z/2.0)/2.0/_math.pi
    if coupled:
        tunes = _np.log(_np.linalg.eigvals(m1turn))/2.0/_math.pi/1j
        tune_x = tunes[_np.argmin(abs(_np.sin(tunes.real)-_math.sin(tune_x)))]
        tune_y = tunes[_np.argmin(abs(_np.sin(tunes.real)-_math.sin(tune_y)))]
        tune_z = tunes[_np.argmin(abs(_np.sin(tunes.real)-_math.sin(tune_z)))]

    return tune_x, tune_y, tune_z, trace_x, trace_y, trace_z, m1turn, \
        closed_orbit


@_interactive
def get_chromaticities(accelerator, energy_offset=1e-6):
    cav_on = accelerator.cavity_on
    rad_on = accelerator.radiation_on
    accelerator.radiation_on = False
    accelerator.cavity_on = False

    nux, nuy, *_ = get_frac_tunes(accelerator, dim='4D', energy_offset=0.0)
    nux_den, nuy_den, *_ = get_frac_tunes(
        accelerator, dim='4D', energy_offset=energy_offset)
    chromx = (nux_den - nux)/energy_offset
    chromy = (nuy_den - nuy)/energy_offset

    accelerator.cavity_on = cav_on
    accelerator.radiation_on = rad_on
    return chromx, chromy


@_interactive
def get_mcf(accelerator, order=1, energy_offset=None):
    """Return momentum compaction factor of the accelerator"""
    if energy_offset is None:
        energy_offset = _np.linspace(-1e-3, 1e-3, 11)

    accel = accelerator[:]
    _tracking.set_4d_tracking(accel)
    leng = accel.length

    dl = _np.zeros(_np.size(energy_offset))
    for i, ene in enumerate(energy_offset):
        cod = _tracking.find_orbit4(accel, ene)
        cod = _np.concatenate([cod.flatten(), [ene, 0]])
        T, *_ = _tracking.ring_pass(accel, cod)
        dl[i] = T[5]/leng

    polynom = _np.polynomial.polynomial.polyfit(energy_offset, dl, order)
    polynom = polynom[1:]
    if len(polynom) == 1:
        polynom = polynom[0]
    return polynom


@_interactive
def get_beam_size(accelerator, coupling=0.0, closed_orbit=None, twiss=None,
                  indices='open'):
    """Return beamsizes (stddev) along ring"""

    # twiss parameters
    twi = twiss
    if twi is None:
        fpnt = closed_orbit if closed_orbit is None else closed_orbit[:, 0]
        twi, *_ = calc_twiss(accelerator, fixed_point=fpnt, indices=indices)

    betax, alphax, etax, etapx = twi.betax, twi.alphax, twi.etax, twi.etapx
    betay, alphay, etay, etapy = twi.betay, twi.alphay, twi.etay, twi.etapy

    gammax = (1.0 + alphax**2)/betax
    gammay = (1.0 + alphay**2)/betay

    # emittances and energy spread
    equi = EquilibriumParameters(accelerator)
    emit0 = equi.emit0
    sigmae = equi.espread0
    emitx = emit0 / (1.0 + coupling)
    emity = emitx * coupling

    # beamsizes per se
    sigmax = _np.sqrt(emitx * betax + (sigmae * etax)**2)
    sigmay = _np.sqrt(emity * betay + (sigmae * etay)**2)
    sigmaxl = _np.sqrt(emitx * gammax + (sigmae * etapx)**2)
    sigmayl = _np.sqrt(emity * gammay + (sigmae * etapy)**2)
    return sigmax, sigmay, sigmaxl, sigmayl, emitx, emity, twi


@_interactive
def get_transverse_acceptance(accelerator, twiss=None, init_twiss=None,
                              fixed_point=None, energy_offset=0.0):
    """Return linear transverse horizontal and vertical physical acceptances"""

    if twiss is None:
        twiss, _ = calc_twiss(
            accelerator, init_twiss=init_twiss, fixed_point=fixed_point,
            indices='closed')
    n_twi = len(twiss)
    n_acc = len(accelerator)
    if n_twi not in {n_acc, n_acc+1}:
        raise OpticsException(
            'Mismatch between size of accelerator and size of twiss object')

    closed_orbit = twiss.co
    betax, betay, etax, etay = twiss.betax, twiss.betay, twiss.etax, twiss.etay

    # physical apertures
    lattice = accelerator.trackcpp_acc.lattice
    hmax = _lattice.get_attribute(accelerator, 'hmax')
    vmax = _lattice.get_attribute(accelerator, 'vmax')
    if len(hmax) != n_twi:
        hmax = _np.append(hmax, hmax[-1])
        vmax = _np.append(vmax, vmax[-1])

    # calcs local linear acceptances
    co_x, co_y = closed_orbit[(0, 2), :]

    # calcs acceptance with beta at entrance of elements
    betax_sqrt, betay_sqrt = _np.sqrt(betax), _np.sqrt(betay)
    accepx_pos = (hmax - (co_x + etax * energy_offset)) / betax_sqrt
    accepx_neg = (hmax + (co_x + etax * energy_offset)) / betax_sqrt
    accepy_pos = (vmax - (co_y + etay * energy_offset)) / betay_sqrt
    accepy_neg = (vmax + (co_y + etay * energy_offset)) / betay_sqrt
    accepx_pos[accepx_pos < 0] = 0
    accepx_neg[accepx_neg < 0] = 0
    accepx_pos[accepy_pos < 0] = 0
    accepx_neg[accepy_neg < 0] = 0
    accepx = _np.min([accepx_pos, accepx_neg], axis=0)
    accepx *= accepx
    accepy = _np.min([accepy_pos, accepy_neg], axis=0)
    accepy *= accepy

    return accepx, accepy, twiss
