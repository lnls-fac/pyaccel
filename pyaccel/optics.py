"""Optics module."""

import math as _math
from os import stat
import numpy as _np
import scipy.linalg as _scylin

import mathphys as _mp
import trackcpp as _trackcpp

from . import lattice as _lattice
from . import tracking as _tracking
from . import accelerator as _accelerator
from .utils import interactive as _interactive


class OpticsException(Exception):
    """."""


class Twiss(_np.record):
    """."""

    DTYPE = '<f8'
    # NOTE: This ordering must be compatible with the one defined in file
    # interface.cpp of repository trackccp, inside the definition of the method
    # calc_twiss_wrapper.
    ORDER = _mp.functions.get_namedtuple(
        'Order', field_names=[
            'spos',
            'betax', 'alphax', 'mux', 'betay', 'alphay', 'muy',
            'etax', 'etapx', 'etay', 'etapy',
            'rx', 'px', 'ry', 'py', 'de', 'dl'])

    def __setattr__(self, attr, val):
        """."""
        if attr == 'co':
            self._set_co(val)
        else:
            super().__setattr__(attr, val)

    def __str__(self):
        """."""
        rst = ''
        rst += 'spos          : '+'{0:+10.3e}'.format(self.spos)
        fmt = '{0:+10.3e}, {1:+10.3e}'
        rst += '\nrx, ry        : '+fmt.format(self.rx, self.ry)
        rst += '\npx, py        : '+fmt.format(self.px, self.py)
        rst += '\nde, dl        : '+fmt.format(self.de, self.dl)
        rst += '\nmux, muy      : '+fmt.format(self.mux, self.muy)
        rst += '\nbetax, betay  : '+fmt.format(self.betax, self.betay)
        rst += '\nalphax, alphay: '+fmt.format(self.alphax, self.alphay)
        rst += '\netax, etapx   : '+fmt.format(self.etax, self.etapx)
        rst += '\netay, etapy   : '+fmt.format(self.etay, self.etapy)
        return rst

    @property
    def co(self):
        """."""
        return _np.array([
            self.rx, self.px, self.ry, self.py, self.de, self.dl])

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
        twi = TwissArray(1)
        cod = kwrgs.get('co', (0.0,)*6)
        twi['rx'], twi['px'], twi['ry'], twi['py'], twi['de'], twi['dl'] = cod
        twi['mux'], twi['muy'] = kwrgs.get('mu', (0.0, 0.0))
        twi['betax'], twi['betay'] = kwrgs.get('beta', (0.0, 0.0))
        twi['alphax'], twi['alphay'] = kwrgs.get('alpha', (0.0, 0.0))
        twi['etax'], twi['etapx'] = kwrgs.get('etax', (0.0, 0.0))
        twi['etay'], twi['etapy'] = kwrgs.get('etay', (0.0, 0.0))
        return twi[0]

    @classmethod
    def from_trackcpp(cls, twi_):
        """Create numpy array from _trackcpp.Twiss object.

        Args:
            twi_ (trackcpp.Twiss): original Twiss object to convert from

        Returns:
            numpy.ndarray: numpy array to serve as buffer for the new Twiss
                object.

        """
        twi = TwissArray(1)
        twi['spos'] = twi_.spos
        twi['betax'] = twi_.betax
        twi['alphax'] = twi_.alphax
        twi['mux'] = twi_.mux
        twi['betay'] = twi_.betay
        twi['alphay'] = twi_.alphay
        twi['muy'] = twi_.muy
        twi['etax'] = twi_.etax[0]
        twi['etapx'] = twi_.etax[1]
        twi['etay'] = twi_.etay[0]
        twi['etapy'] = twi_.etay[1]
        twi['rx'] = twi_.co.rx
        twi['px'] = twi_.co.px
        twi['ry'] = twi_.co.ry
        twi['py'] = twi_.co.py
        twi['de'] = twi_.co.de
        twi['dl'] = twi_.co.dl
        return twi[0]

    def to_trackcpp(self):
        """."""
        twi = _trackcpp.Twiss()
        twi.spos = float(self.spos)
        twi.betax = float(self.betax)
        twi.alphax = float(self.alphax)
        twi.mux = float(self.mux)
        twi.betay = float(self.betay)
        twi.alphay = float(self.alphay)
        twi.muy = float(self.muy)
        twi.etax[0] = float(self.etax)
        twi.etax[1] = float(self.etapx)
        twi.etay[0] = float(self.etay)
        twi.etay[1] = float(self.etapy)
        twi.co.rx = float(self.rx)
        twi.co.px = float(self.px)
        twi.co.ry = float(self.ry)
        twi.co.py = float(self.py)
        twi.co.de = float(self.de)
        twi.co.dl = float(self.dl)
        return twi

    def _set_co(self, value):
        """."""
        try:
            leng = len(value)
        except TypeError:
            leng = 6
            value = [value, ]*leng
        if leng != 6:
            raise ValueError('closed orbit must have 6 elements.')
        self[Twiss.ORDER.rx], self[Twiss.ORDER.px] = value[0], value[1]
        self[Twiss.ORDER.ry], self[Twiss.ORDER.py] = value[2], value[3]
        self[Twiss.ORDER.de], self[Twiss.ORDER.dl] = value[4], value[5]


class TwissArray(_np.ndarray):
    """."""

    def __eq__(self, other):
        """."""
        return _np.all(super().__eq__(other))

    def __new__(cls, twiss=None, copy=True):
        """."""
        length = 1
        if isinstance(twiss, (int, _np.int)):
            length = twiss
            twiss = None
        elif isinstance(twiss, TwissArray):
            return twiss.copy() if copy else twiss

        if twiss is None:
            arr = _np.zeros((length, len(Twiss.ORDER)), dtype=Twiss.DTYPE)
        elif isinstance(twiss, _np.ndarray):
            arr = twiss.copy() if copy else twiss
        elif isinstance(twiss, _np.record):
            arr = _np.ndarray(
                (twiss.size, len(Twiss.ORDER)), buffer=twiss.data)
            arr = arr.copy() if copy else arr
        elif isinstance(twiss, _trackcpp.Twiss):
            arr = Twiss.from_trackcpp(twiss)
        elif isinstance(twiss, _trackcpp.CppTwissVector):
            arr = _np.zeros((twiss.size, len(Twiss.ORDER)), dtype=Twiss.DTYPE)
            for i in range(len(twiss)):
                arr[:, i] = Twiss.from_trackcpp(twiss[i])

        fmts = [(fmt, Twiss.DTYPE) for fmt in Twiss.ORDER._fields]
        return super().__new__(
            cls, shape=(arr.shape[0], ), dtype=(Twiss, fmts), buffer=arr)

    @property
    def spos(self):
        """."""
        return self['spos']

    @spos.setter
    def spos(self, value):
        self['spos'] = value

    @property
    def betax(self):
        """."""
        return self['betax']

    @betax.setter
    def betax(self, value):
        self['betax'] = value

    @property
    def alphax(self):
        """."""
        return self['alphax']

    @alphax.setter
    def alphax(self, value):
        self['alphax'] = value

    @property
    def mux(self):
        """."""
        return self['mux']

    @mux.setter
    def mux(self, value):
        self['mux'] = value

    @property
    def betay(self):
        """."""
        return self['betay']

    @betay.setter
    def betay(self, value):
        self['betay'] = value

    @property
    def alphay(self):
        """."""
        return self['alphay']

    @alphay.setter
    def alphay(self, value):
        self['alphay'] = value

    @property
    def muy(self):
        """."""
        return self['muy']

    @muy.setter
    def muy(self, value):
        self['muy'] = value

    @property
    def etax(self):
        """."""
        return self['etax']

    @etax.setter
    def etax(self, value):
        self['etax'] = value

    @property
    def etapx(self):
        """."""
        return self['etapx']

    @etapx.setter
    def etapx(self, value):
        self['etapx'] = value

    @property
    def etay(self):
        """."""
        return self['etay']

    @etay.setter
    def etay(self, value):
        self['etay'] = value

    @property
    def etapy(self):
        """."""
        return self['etapy']

    @etapy.setter
    def etapy(self, value):
        self['etapy'] = value

    @property
    def rx(self):
        """."""
        return self['rx']

    @rx.setter
    def rx(self, value):
        self['rx'] = value

    @property
    def px(self):
        """."""
        return self['px']

    @px.setter
    def px(self, value):
        self['px'] = value

    @property
    def ry(self):
        """."""
        return self['ry']

    @ry.setter
    def ry(self, value):
        self['ry'] = value

    @property
    def py(self):
        """."""
        return self['py']

    @py.setter
    def py(self, value):
        self['py'] = value

    @property
    def de(self):
        """."""
        return self['de']

    @de.setter
    def de(self, value):
        self['de'] = value

    @property
    def dl(self):
        """."""
        return self['dl']

    @dl.setter
    def dl(self, value):
        self['dl'] = value

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

    def to_trackcpp(self):
        """Convert Twiss object to appropriate trackcpp object.

        Returns:
            _trackcpp.Twiss() | _trackcpp.CppTwissVector: If self.size == 1
                the return value will be a `_trackcpp.Twiss` object. Otherwise
                a `_trackcpp.CppTwissVector` will be returned.

        """
        twi = _trackcpp.CppTwissVector()
        for twi_ in self:
            twi.push_back(twi_.to_trackcpp())

        if self.size == 1:
            twi = twi.back()
        return twi

    @staticmethod
    def compose(twiss_list):
        """."""
        if isinstance(twiss_list, (list, tuple)):
            for val in twiss_list:
                if not isinstance(val, (Twiss, TwissArray)):
                    raise OpticsException(
                        'can only compose lists of Twiss objects.')
        else:
            raise OpticsException('can only compose lists of Twiss objects.')

        arrs = list()
        for val in twiss_list:
            arrs.append(_np.ndarray(
                (val.size, len(Twiss.ORDER)), buffer=val.data))
        arrs = _np.vstack(arrs)
        return TwissArray(arrs)


@_interactive
def calc_twiss(
        accelerator=None, init_twiss=None, fixed_point=None,
        indices='open', energy_offset=None):
    """Return Twiss parameters of uncoupled dynamics.

    Args:
        accelerator (Accelerator, optional): Defaults to None.
        init_twiss (Twiss, optional): Twiss parameters at the start of first
            element. Defaults to None.
        fixed_point (numpy.ndarray, optional): 6D position at the start of
            first element. Defaults to None.
        indices (str, optional): 'open' or 'closed'. Defaults to 'open'.
        energy_offset (float, optional): float denoting the energy deviation
            (used only for periodic solutions). Defaults to None.

    Raises:
        pyaccel.tracking.TrackingException: When find_orbit fails to converge.
        OpticsException: When trackcpp.calc_twiss fails,  or when accelerator
            is not configured properly.

    Returns:
        Twiss: object (closed orbit data is in the objects vector)
        numpy.ndarray: one-turn transfer matrix

    """
    indices = _tracking._process_indices(accelerator, indices)

    _m66 = _trackcpp.Matrix()
    twiss = _np.zeros((len(accelerator)+1, len(Twiss.ORDER)), dtype=_np.float)

    if init_twiss is not None:
        # as a transport line: uses init_twiss
        _init_twiss = init_twiss.to_trackcpp()
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

    r = _trackcpp.calc_twiss_wrapper(
        accelerator.trackcpp_acc, _fixed_point, _m66, twiss, _init_twiss)
    if r > 0:
        raise OpticsException(_trackcpp.string_error_messages[r])

    twiss = TwissArray(twiss, copy=False)
    m66 = _tracking._CppMatrix2Numpy(_m66)

    return twiss[indices], m66


class EquilibriumParametersIntegrals:
    """."""

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

        ints = 'Jx,Jy,Je'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'taux,tauy,taue'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [ms]')
        rst += ', '.join([fmtn.format(1000*getattr(self, x)) for x in ints])

        ints = 'alphax,alphay,alphae'.split(',')
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
        E0 = self._acc.energy / 1e9  # in GeV
        E0 *= (1 + self._energy_offset)
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jx / leng

    @property
    def alphay(self):
        """."""
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        E0 *= (1 + self._energy_offset)
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jy / leng

    @property
    def alphae(self):
        """."""
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
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
        """."""
        E0 = self._acc.energy / 1e9  # in GeV
        E0 *= (1 + self._energy_offset)
        rad_cgamma = _mp.constants.rad_cgamma
        return rad_cgamma/(2*_math.pi) * E0**4 * self.I2 * 1e9  # in eV

    @property
    def overvoltage(self):
        """."""
        v_cav = get_rf_voltage(self._acc)
        return v_cav/self.U0

    @property
    def syncphase(self):
        """."""
        return _math.pi - _math.asin(1/self.overvoltage)

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
        v_cav = get_rf_voltage(self._acc)
        harmon = self._acc.harmonic_number
        return _math.sqrt(
            self.etac*harmon*v_cav*_math.cos(self.syncphase)/(2*_math.pi*E0))

    @property
    def bunlen(self):
        """."""
        vel = self._acc.velocity
        rev_freq = get_revolution_frequency(self._acc)

        bunlen = vel * abs(self.etac) * self.espread0
        bunlen /= 2*_math.pi * self.synctune * rev_freq
        return bunlen

    @property
    def rf_acceptance(self):
        """."""
        E0 = self._acc.energy
        sph = self.syncphase
        V = get_rf_voltage(self._acc)
        ov = self.overvoltage
        h = self._acc.harmonic_number
        etac = self.etac

        eaccep2 = V * _math.sin(sph) / (_math.pi*h*abs(etac)*E0)
        eaccep2 *= 2 * (_math.sqrt(ov**2 - 1.0) - _math.acos(1.0/ov))
        return _math.sqrt(eaccep2)

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
        return _np.sqrt(emitx*self._twi.alphax + (espread0*self._twi.etapx)**2)

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
        return _np.sqrt(emity*self._twi.alphay + (espread0*self._twi.etapy)**2)

    def as_dict(self):
        """."""
        pars = {
            'energy_offset', 'twiss',
            'I1x', 'I2', 'I3', 'I3a', 'I4x', 'I5x', 'I6x',
            'I1y', 'I4y', 'I5y', 'I6y',
            'Jx', 'Jy', 'Je',
            'alphax', 'alphay', 'alphae',
            'taux', 'tauy', 'taue',
            'espread0',
            'emitx', 'emity', 'emit0',
            'bunch_length',
            'U0', 'overvoltage', 'syncphase', 'synctune',
            'alpha', 'etac', 'rf_acceptance',
            }
        dic = {par: getattr(self, par) for par in pars}
        dic['energy'] = self.accelerator.energy
        return dic

    def _calc_radiation_integrals(self):
        """Calculate radiation integrals for periodic systems."""
        acc = self._acc
        twi, m66 = calc_twiss(
            acc, indices='closed', energy_offset=self._energy_offset)
        self._twi = twi
        self._m66 = m66
        self._alpha = get_mcf(acc, energy_offset=self._energy_offset)

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

        Hx_in = get_curlyh(betax_in, alphax_in, etax_in, etapx_in)
        Hx_out = get_curlyh(betax_out, alphax_out, etax_in, etapx_out)

        Hy_in = get_curlyh(betay_in, alphay_in, etay_in, etapy_in)
        Hy_out = get_curlyh(betay_out, alphay_out, etay_in, etapy_out)

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


class EquilibriumParametersOhmiFormalism:
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
        v_cav = get_rf_voltage(self._acc)
        return v_cav/self.U0

    @property
    def syncphase(self):
        """."""
        return _math.pi - _math.asin(1/self.overvoltage)

    @property
    def etac(self):
        """."""
        vel = self._acc.velocity
        rev_freq = get_revolution_frequency(self._acc)

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
        V = get_rf_voltage(self._acc)
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
            calc_ohmienvelope(
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
        alphas = -_np.log(alphas) * get_revolution_frequency(self._acc)

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
def calc_ohmienvelope(
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
            accelerator, indices='closed', closed_orbit=fixed_point)

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


@_interactive
def calc_emittance_coupling(
        accelerator, mode='fitting', x0=1e-5, y0=1e-8, nr_turns=100):
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
def get_revolution_frequency(accelerator):
    """."""
    return get_rf_frequency(accelerator) / accelerator.harmonic_number


@_interactive
def get_revolution_period(accelerator):
    """."""
    return 1 / get_revolution_frequency(accelerator)


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
    """."""
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
def get_mcf(accelerator, order=1, energy_offset=None, energy_range=None):
    """Return momentum compaction factor of the accelerator"""
    if energy_range is None:
        energy_range = _np.linspace(-1e-3, 1e-3, 11)

    if energy_offset is not None:
        energy_range += energy_offset

    accel = accelerator[:]
    _tracking.set_4d_tracking(accel)
    leng = accel.length

    dl = _np.zeros(_np.size(energy_range))
    for i, ene in enumerate(energy_range):
        cod = _tracking.find_orbit4(accel, ene)
        cod = _np.concatenate([cod.flatten(), [ene, 0]])
        T, *_ = _tracking.ring_pass(accel, cod)
        dl[i] = T[5]/leng

    polynom = _np.polynomial.polynomial.polyfit(energy_range, dl, order)
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
    equi = EquilibriumParametersIntegrals(accelerator)
    sigmae = equi.espread0
    emitx = (equi.emitx + equi.emity*coupling) / (1.0 + coupling)
    emity = (equi.emity + equi.emitx*coupling) / (1.0 + coupling)

    # beamsizes per se
    sigmax = _np.sqrt(emitx * betax + (sigmae * etax)**2)
    sigmay = _np.sqrt(emity * betay + (sigmae * etay)**2)
    sigmaxl = _np.sqrt(emitx * gammax + (sigmae * etapx)**2)
    sigmayl = _np.sqrt(emity * gammay + (sigmae * etapy)**2)
    return sigmax, sigmay, sigmaxl, sigmayl, emitx, emity, twi


@_interactive
def calc_transverse_acceptance(
        accelerator, twiss=None, init_twiss=None, fixed_point=None,
        energy_offset=0.0):
    """Return transverse horizontal and vertical physical acceptances."""
    if twiss is None:
        twiss, _ = calc_twiss(
            accelerator, init_twiss=init_twiss, fixed_point=fixed_point,
            indices='closed', energy_offset=energy_offset)

    n_twi = len(twiss)
    n_acc = len(accelerator)
    if n_twi not in {n_acc, n_acc+1}:
        raise OpticsException(
            'Mismatch between size of accelerator and size of twiss object')

    betax = twiss.betax
    betay = twiss.betay
    co_x = twiss.rx
    co_y = twiss.ry

    # physical apertures
    hmax, hmin = _np.zeros(n_twi), _np.zeros(n_twi)
    vmax, vmin = _np.zeros(n_twi), _np.zeros(n_twi)
    for idx, ele in enumerate(accelerator):
        hmax[idx], hmin[idx] = ele.hmax, ele.hmin
        vmax[idx], vmin[idx] = ele.vmax, ele.vmin
    if n_twi > n_acc:
        hmax[-1], hmin[-1] = hmax[0], hmin[0]
        vmax[-1], vmin[-1] = vmax[0], vmin[0]

    # calcs acceptance with beta at entrance of elements
    betax_sqrt, betay_sqrt = _np.sqrt(betax), _np.sqrt(betay)
    accepx_pos = (hmax - co_x)
    accepx_neg = (-hmin + co_x)
    accepy_pos = (vmax - co_y)
    accepy_neg = (-vmin + co_y)
    accepx_pos /= betax_sqrt
    accepx_neg /= betax_sqrt
    accepy_pos /= betay_sqrt
    accepy_neg /= betay_sqrt
    accepx = _np.minimum(accepx_pos, accepx_neg)
    accepy = _np.minimum(accepy_pos, accepy_neg)
    accepx[accepx < 0] = 0
    accepy[accepy < 0] = 0
    accepx *= accepx
    accepy *= accepy
    return accepx, accepy, twiss


@_interactive
def calc_tousheck_energy_acceptance(
        accelerator, energy_offsets=None, track=False, check_tune=False,
        **kwargs):
    """."""
    vcham_sts = accelerator.vchamber_on
    rad_sts = accelerator.radiation_on
    cav_sts = accelerator.cavity_on

    accelerator.radiation_on = False
    accelerator.cavity_on = False
    accelerator.vchamber_on = False

    if energy_offsets is None:
        energy_offsets = _np.linspace(1e-6, 6e-2, 60)

    if _np.any(energy_offsets < 0):
        raise ValueError('delta must be a positive vector.')

    # ############ Calculate physical aperture ############

    hmax = _lattice.get_attribute(accelerator, 'hmax', indices='closed')
    hmin = _lattice.get_attribute(accelerator, 'hmin', indices='closed')
    twi0, *_ = calc_twiss(accelerator, indices='closed')
    tune0 = _np.array([twi0[-1].mux, twi0[-1].muy]) / (2*_np.pi)
    rx0 = twi0.rx
    px0 = twi0.px

    # positive energies
    curh_pos, ap_phys_pos, tune_pos, beta_pos = _calc_phys_apert_for_tousheck(
        accelerator, energy_offsets, rx0, px0, hmax, hmin)
    # negative energies
    curh_neg, ap_phys_neg, tune_neg, beta_neg = _calc_phys_apert_for_tousheck(
        accelerator, -energy_offsets, rx0, px0, hmax, hmin)

    # Considering synchrotron oscillations, negative energy deviations will
    # turn into positive ones and vice-versa, so the apperture must be
    # symmetric:
    ap_phys = _np.minimum(ap_phys_pos, ap_phys_neg)

    # ############ Calculate Dynamic Aperture ############
    ap_dyn_pos = _np.full(energy_offsets.shape, _np.inf)
    ap_dyn_neg = ap_dyn_pos.copy()
    if track:
        nturns = kwargs.get('track_nr_turns', 131)
        parallel = kwargs.get('track_parallel', True)
        curh_track = kwargs.get(
            'track_curh', _np.linspace(0, 4e-6, 30))
        ener_pos = kwargs.get(
            'track_delta_pos', _np.linspace(0.02, energy_offsets.max(), 20))
        ener_neg = kwargs.get('track_delta_neg', -ener_pos)

        beta_pos = _np.interp(ener_pos, energy_offsets, beta_pos)
        ap_dyn_pos = _calc_dyn_apert_for_tousheck(
            accelerator, ener_pos, curh_track, beta_pos, nturns,
            parallel=parallel)
        ap_dyn_pos = _np.interp(energy_offsets, ener_pos, ap_dyn_pos)

        beta_neg = _np.interp(-ener_neg, energy_offsets, beta_neg)
        ap_dyn_neg = _calc_dyn_apert_for_tousheck(
            accelerator, ener_neg, curh_track, beta_neg, nturns,
            parallel=parallel)
        ap_dyn_neg = _np.interp(energy_offsets, -ener_neg, ap_dyn_neg)

    accelerator.vchamber_on = vcham_sts
    accelerator.radiation_on = rad_sts
    accelerator.cavity_on = cav_sts

    # ############ Check tunes ############
    # Make sure tunes don't cross int and half-int resonances
    # Must be symmetric due to syncrhotron oscillations
    ap_tune = _np.full(energy_offsets.shape, _np.inf)
    if check_tune:
        tune0_int = _np.floor(tune0)
        tune_pos -= tune0_int[:, None]
        tune_neg -= tune0_int[:, None]
        tune0 -= tune0_int

        # make M*tune > b to check for quadrant crossing:
        quadrant0 = _np.array(tune0 > 0.5, dtype=float)
        M = _np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        b = _np.array([0, 0, -1/2, -1/2])
        b += _np.r_[quadrant0/2, -quadrant0/2]

        boo_pos = _np.dot(M, tune_pos) > b[:, None]
        boo_neg = _np.dot(M, tune_neg) > b[:, None]
        nok_pos = ~boo_pos.all(axis=0)
        nok_neg = ~boo_neg.all(axis=0)

        ap_tune[nok_pos] = 0
        ap_tune[nok_neg] = 0

    # ############ Calculate aperture ############
    ap_pos = _np.minimum(ap_dyn_pos, ap_phys, ap_tune)
    ap_neg = _np.minimum(ap_dyn_neg, ap_phys, ap_tune)
    # make sure it is monotonic with energy:
    for idx in _np.arange(1, ap_pos.size):
        ap_pos[idx] = _np.minimum(ap_pos[idx], ap_pos[idx-1])
    for idx in _np.arange(1, ap_neg.size):
        ap_neg[idx] = _np.minimum(ap_neg[idx], ap_neg[idx-1])

    # ############ Calculate energy acceptance along ring ############
    comp = curh_pos >= ap_pos[:, None]
    idcs = _np.argmax(comp, axis=0)
    boo = _np.take_along_axis(comp, _np.expand_dims(idcs, axis=0), axis=0)
    idcs[~boo.ravel()] = ap_pos.size-1
    accep_pos = energy_offsets[idcs]

    comp = curh_neg >= ap_neg[:, None]
    idcs = _np.argmax(comp, axis=0)
    boo = _np.take_along_axis(comp, _np.expand_dims(idcs, axis=0), axis=0)
    idcs[~boo.ravel()] = ap_neg.size-1
    accep_neg = -energy_offsets[idcs]

    return accep_neg, accep_pos


@_interactive
def get_curlyh(beta, alpha, x, xl):
    """."""
    gamma = (1 + alpha*alpha) / beta
    return beta*xl*xl + 2*alpha*x*xl + gamma*x*x


def _calc_phys_apert_for_tousheck(
        accelerator, energy_offsets, rx0, px0, hmax, hmin):
    curh = _np.full((energy_offsets.size, rx0.size), _np.inf)
    tune = _np.full((2, energy_offsets.size), _np.nan)
    ap_phys = _np.zeros(energy_offsets.size)
    beta = _np.ones(energy_offsets.size)
    try:
        for idx, delta in enumerate(energy_offsets):
            twi, *_ = calc_twiss(
                accelerator, energy_offset=delta, indices='closed')
            if _np.any(_np.isnan(twi[0].betax)):
                raise OpticsException('error')
            tune[0, idx] = twi[-1].mux / (2*_np.pi)
            tune[1, idx] = twi[-1].muy / (2*_np.pi)
            beta[idx] = twi[0].betax
            rx = twi.rx
            px = twi.px
            betax = twi.betax
            dcox = rx - rx0
            dcoxp = px - px0
            curh[idx] = get_curlyh(betax, twi.alphax, dcox, dcoxp)

            apper_loc = _np.minimum((hmax - rx)**2, (hmin + rx)**2)
            ap_phys[idx] = _np.min(apper_loc / betax)
    except (OpticsException, _tracking.TrackingException):
        pass
    return curh, ap_phys, tune, beta


def _calc_dyn_apert_for_tousheck(
        accelerator, energies, curh, beta, nturns, parallel=False):
    accelerator.cavity_on = False
    accelerator.radiation_on = False
    accelerator.vchamber_on = False

    rin = _np.full((6, energies.size, curh.size), _np.nan)
    try:
        for idx, en in enumerate(energies):
            rin[:4, idx, :] = _tracking.find_orbit4(
                accelerator, energy_offset=en).ravel()[:, None]
    except _tracking.TrackingException:
        pass
    rin = rin.reshape(6, -1)

    accelerator.cavity_on = True
    accelerator.radiation_on = True
    accelerator.vchamber_on = True
    orb6d = _tracking.find_orbit6(accelerator)

    # Track positive energies
    curh0, ener = _np.meshgrid(curh, energies)
    xl = _np.sqrt(curh0/beta[:, None])

    rin[1, :] += xl.ravel()
    rin[2, :] += 1e-6
    rin[4, :] = orb6d[4] + ener.ravel()
    rin[5, :] = orb6d[5]

    _, _, lostturn, *_ = _tracking.ring_pass(
        accelerator, rin, nturns, turn_by_turn=False, parallel=parallel)
    lostturn = _np.reshape(lostturn, curh0.shape)
    lost = lostturn != nturns

    ind_dyn = _np.argmax(lost, axis=1)
    ap_dyn = curh[ind_dyn]
    return ap_dyn


def _sandwich_matrix(mat1, mat2):
    """."""
    # return mat1 @ mat2 @ mat1.swapaxes(-1, -2)
    return _np.dot(mat1, _np.dot(mat2, mat1.T))
