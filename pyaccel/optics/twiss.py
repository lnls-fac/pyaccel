"""Twiss Module."""

import mathphys as _mp
import numpy as _np
import trackcpp as _trackcpp

from .. import tracking as _tracking
from ..utils import interactive as _interactive
from .miscellaneous import OpticsError as _OpticsError


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
        if isinstance(twiss, int):
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
    def gammax(self):
        """."""
        return (1 + self['alphax']*self['alphax'])/self['betax']

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
    def gammay(self):
        """."""
        return (1 + self['alphay']*self['alphay'])/self['betay']

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
                    raise _OpticsError(
                        'can only compose lists of Twiss objects.')
        else:
            raise _OpticsError('can only compose lists of Twiss objects.')

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
        pyaccel.tracking.TrackingError: When find_orbit fails to converge.
        pyaccel.optics.OpticsError: When trackcpp.calc_twiss fails, or
            when accelerator is not configured properly.

    Returns:
        Twiss: object (closed orbit data is in the objects vector)
        numpy.ndarray: one-turn transfer matrix

    """
    indices = _tracking._process_indices(accelerator, indices)

    _m66 = _trackcpp.Matrix()
    twiss = _np.zeros((len(accelerator)+1, len(Twiss.ORDER)), dtype=float)

    if init_twiss is not None:
        # as a transport line: uses init_twiss
        _init_twiss = init_twiss.to_trackcpp()
        if fixed_point is None:
            _fixed_point = _init_twiss.co
        else:
            raise _OpticsError(
                'arguments init_twiss and fixed_point are mutually exclusive')
    else:
        # as a periodic system: try to find periodic solution
        if accelerator.harmonic_number == 0:
            raise _OpticsError(
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
                raise _OpticsError(
                    'The radiation is on but the cavity is off')
            else:
                r = _trackcpp.track_findorbit6(
                    accelerator.trackcpp_acc, _closed_orbit,
                    _fixed_point_guess)

            if r > 0:
                raise _tracking.TrackingError(
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
        raise _OpticsError(_trackcpp.string_error_messages[r])

    twiss = TwissArray(twiss, copy=False)
    m66 = _np.array(_m66)

    return twiss[indices], m66
