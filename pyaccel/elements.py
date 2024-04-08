
# -*- coding: utf-8 -*-

import warnings as _warnings
import numpy as _numpy
from .utils import interactive as _interactive, Polynom as _Polynom
from . import get_backend
backend = get_backend()
backend_package_name = {"cpp":"trackcpp", "julia":"Track.Elements"}

_ELEM_ATTRIBUTES = [
    'angle',
    'angle_in',
    'angle_out',
    'fam_name',
    'fint_in',
    'fint_out',
    'frequency',
    'gap',
    'hkick',
    'hmax',
    'hmin',
    'kickmap',
    'kicktable_idx',
    'length',
    'matrix66',
    'nr_steps',
    'pass_method',
    'phase_lag',
    'polynom_a',
    'polynom_b',
    'r_in',
    'r_out',
    'rescale_kicks',
    't_in',
    't_out',
    'thin_KL',
    'thin_SL',
    'vchamber',
    'vkick',
    'vmax',
    'vmin',
    'voltage'
]


@_interactive
def marker(fam_name):
    """Create a marker element.

    Keyword arguments:
    fam_name -- family name
    """
    ele = backend.marker(fam_name)
    return Element(element=ele)


@_interactive
def bpm(fam_name):
    """Create a beam position monitor element.

    Keyword arguments:
    fam_name -- family name
    """
    ele = backend.bpm(fam_name)
    return Element(element=ele)


@_interactive
def drift(fam_name, length):
    """Create a drift element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    """
    ele = backend.drift(fam_name, length)
    return Element(element=ele)


@_interactive
def matrix(fam_name, length):
    """Create a matrix element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    """
    ele = backend.matrix(fam_name, length)
    return Element(element=ele)


@_interactive
def hcorrector(fam_name, length=0.0, hkick=0.0):
    """Create a horizontal corrector element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    hkick -- horizontal kick [rad]
    """
    ele = backend.hcorrector(fam_name, length, hkick)
    return Element(element=ele)


@_interactive
def vcorrector(fam_name, length=0.0, vkick=0.0):
    """Create a vertical corrector element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    vkick -- vertical kick [rad]
    """
    ele = backend.vcorrector(fam_name, length, vkick)
    return Element(element=ele)


@_interactive
def corrector(fam_name, length=0.0, hkick=0.0, vkick=0.0):
    """Create a corrector element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    hkick -- horizontal kick [rad]
    vkick -- vertical kick [rad]
    """
    ele = backend.corrector(fam_name, length, hkick, vkick)
    return Element(element=ele)


@_interactive
def rbend(fam_name, length, angle, angle_in=0.0, angle_out=0.0,
          gap=0.0, fint_in=0.0, fint_out=0.0, polynom_a=None,
          polynom_b=None, K=None, S=None):
    """Create a rectangular dipole element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    angle -- [rad]
    angle_in -- [rad]
    angle_out -- [rad]
    K -- [m^-2]
    S -- [m^-3]
    """
    polynom_a, polynom_b = _process_polynoms(polynom_a, polynom_b)
    if K is None:
        K = polynom_b[1]
    if S is None:
        S = polynom_b[2]
    ele = backend.rbend(
        fam_name, length, angle, angle_in, angle_out, gap, fint_in, fint_out,
        polynom_a, polynom_b, K, S)
    return Element(element=ele)


@_interactive
def quadrupole(fam_name, length, K, nr_steps=10):
    """Create a quadrupole element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    K -- [m^-2]
    nr_steps -- number of steps (default 10)
    """
    ele = backend.quadrupole(fam_name, length, K, nr_steps)
    return Element(element=ele)


@_interactive
def sextupole(fam_name, length, S, nr_steps=5):
    """Create a sextupole element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    S -- (1/2!)(d^2By/dx^2)/(Brho)[m^-3]
    nr_steps -- number of steps (default 5)
    """
    e = backend.sextupole(fam_name, length, S, nr_steps)
    return Element(element=e)


@_interactive
def rfcavity(fam_name, length, voltage, frequency, phase_lag=0.0):
    """Create a RF cavity element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    voltage -- [V]
    frequency -- [Hz]
    phase_lag -- [rad]
    """
    ele = backend.rfcavity(
        fam_name, length, frequency, voltage, phase_lag)
    return Element(element=ele)


@_interactive
def kickmap(
        fam_name, kicktable_fname, nr_steps=20,
        rescale_length=1.0, rescale_kicks=1.0):
    """Create a kickmap element.

    Keyword arguments:
    fam_name -- family name
    kicktable_fname -- filename of kicktable
    nr_steps -- number of steps (default 20)
    rescale_length -- rescale kicktable length (default 1)
    rescale_kicks -- rescale all kicktable length (default 1)
    """
    e = backend.kickmap(
        fam_name, kicktable_fname, nr_steps, rescale_length, rescale_kicks)
    return Element(element=e)


def _process_polynoms(polya, polyb):
    # Make sure polya and polyb have same size and are initialized
    if polya is None:
        polya = [0.0, 0.0, 0.0]
    if polyb is None:
        polyb = [0.0, 0.0, 0.0]
    nmax = max(3, len(polya), len(polyb))
    pan = nmax * [0.0]
    pbn = nmax * [0.0]
    for i, pa_ in enumerate(polya):
        pan[i] = pa_
    for i, pb_ in enumerate(polyb):
        pbn[i] = pb_
    return pan, pbn


# @_interactive
# class Kicktable:
#     """."""

#     kicktable_list = _trackcpp.cvar.kicktable_list  # trackcpp vector with all Kicktables in use.

#     def __init__(self, **kwargs):
#         """."""
#         # get kicktable filename
#         if 'kicktable' in kwargs:
#             filename = kwargs['kicktable'].filename
#         elif 'filename' in kwargs:
#             filename = kwargs['filename']
#         else:
#             raise NotImplementedError('Invalid Kicktable argument')

#         # add new kicktable to list or retrieve index of existing one with same fname.
#         idx = _trackcpp.add_kicktable(filename)

#         # update object attributes
#         if idx != -1:
#             self._status = _trackcpp.Status.success
#             self._kicktable_idx = idx
#             self._kicktable = _trackcpp.cvar.kicktable_list[idx]
#         else:
#             self._status = _trackcpp.Status.file_not_found
#             self._kicktable_idx = -1
#             self._kicktable = None

#     @property
#     def trackcpp_kickmap(self):
#         """Return trackcpp Kickmap object."""
#         return self._kicktable

#     @property
#     def kicktable_idx(self):
#         """Return kicktable index in trackcpp kicktable_list vector."""
#         return self._kicktable_idx

#     @property
#     def filename(self):
#         """Filename corresponding to kicktable"""
#         return self._kicktable.filename

#     @property
#     def length(self):
#         """."""
#         return self._kicktable.length

#     @property
#     def x_min(self):
#         """."""
#         return self._kicktable.x_min

#     @property
#     def x_max(self):
#         """."""
#         return self._kicktable.x_max

#     @property
#     def y_min(self):
#         """."""
#         return self._kicktable.y_min

#     @property
#     def y_max(self):
#         """."""
#         return self._kicktable.y_max

#     @property
#     def x_nrpts(self):
#         """."""
#         return self._kicktable.x_nrpts

#     @property
#     def y_nrpts(self):
#         """."""
#         return self._kicktable.y_nrpts

#     @property
#     def status(self):
#         """Return last object status."""
#         return self._status

#     def get_kicks(self, rx, ry):
#         """Return (hkick, vkick) at (rx,ry)."""
#         idx = self.kicktable_idx
#         self._status, hkick, vkick = _trackcpp.kicktable_getkicks_wrapper(idx, rx, ry)
#         return hkick, vkick

#     def __eq__(self, other):
#         """."""
#         if not isinstance(other, Kicktable):
#             return NotImplemented
#         for attr in self._kicktable.__swig_getmethods__:
#             if getattr(self, attr) != getattr(other, attr):
#                 return False
#         return True

class Element:
    """."""

    _t_valid_types = (list, _numpy.ndarray)
    _r_valid_types = (_numpy.ndarray, )

    def __init__(self, element=None, fam_name='', length=0.0):
        """."""

        self._backend_type = backend.backend_type

        if element is None:
            element = backend.Element(fam_name)
            element.length = length
        elif isinstance(element, Element):
            element = element.backend_e
        elif backend.bkd_isinstance(element, backend.Element):
            pass
        else:
            raise TypeError(
                f'element must be a {backend_package_name[backend.backend_type]}.Element or a Element object.')
        self.backend_e = element

    @property
    def fam_name(self):
        """."""
        return self.backend_e.fam_name

    @fam_name.setter
    def fam_name(self, value):
        """."""
        self.backend_e.fam_name = value

    @property
    def pass_method(self):
        """."""
        return backend.PASS_METHODS[
            backend.Int(self.backend_e.pass_method)
            ]

    @pass_method.setter
    def pass_method(self, value):
        """."""
        if isinstance(value, str):
            if value not in backend.PASS_METHODS:
                raise ValueError("pass method '" + value + "' not found")
            else:
                self.backend_e.pass_method = backend.PassMethod(
                    backend.PASS_METHODS.index(value)
                    )
        elif isinstance(value, int):
            if not 0 <= value < len(backend.PASS_METHODS):
                raise IndexError("pass method index out of range")
            self.backend_e.pass_method = backend.PassMethod(value)
        else:
            raise TypeError("pass method value must be string or index")

    @property
    def length(self):
        """."""
        return self.backend_e.length

    @length.setter
    def length(self, value):
        """."""
        self.backend_e.length = value

    @property
    def nr_steps(self):
        """."""
        return self.backend_e.nr_steps

    @nr_steps.setter
    def nr_steps(self, value):
        """."""
        self.backend_e.nr_steps = value

    @property
    def hkick(self):
        """."""
        return self.backend_e.hkick

    @hkick.setter
    def hkick(self, value):
        """."""
        self.backend_e.hkick = value

    @property
    def vkick(self):
        """."""
        return self.backend_e.vkick

    @vkick.setter
    def vkick(self, value):
        """."""
        self.backend_e.vkick = value

    @property
    def angle(self):
        """."""
        return self.backend_e.angle

    @angle.setter
    def angle(self, value):
        """."""
        self.backend_e.angle = value

    @property
    def angle_in(self):
        """."""
        return self.backend_e.angle_in

    @angle_in.setter
    def angle_in(self, value):
        """."""
        self.backend_e.angle_in = value

    @property
    def angle_out(self):
        """."""
        return self.backend_e.angle_out

    @angle_out.setter
    def angle_out(self, value):
        """."""
        self.backend_e.angle_out = value

    @property
    def gap(self):
        """."""
        return self.backend_e.gap

    @gap.setter
    def gap(self, value):
        """."""
        self.backend_e.gap = value

    @property
    def fint_in(self):
        """."""
        return self.backend_e.fint_in

    @fint_in.setter
    def fint_in(self, value):
        """."""
        self.backend_e.fint_in = value

    @property
    def fint_out(self):
        """."""
        return self.backend_e.fint_out

    @fint_out.setter
    def fint_out(self, value):
        """."""
        self.backend_e.fint_out = value

    @property
    def thin_KL(self):
        """."""
        return self.backend_e.thin_KL

    @thin_KL.setter
    def thin_KL(self, value):
        """."""
        self.backend_e.thin_KL = value

    @property
    def thin_SL(self):
        """."""
        return self.backend_e.thin_SL

    @thin_SL.setter
    def thin_SL(self, value):
        """."""
        self.backend_e.thin_SL = value

    @property
    def frequency(self):
        """."""
        return self.backend_e.frequency

    @frequency.setter
    def frequency(self, value):
        """."""
        self.backend_e.frequency = value

    @property
    def voltage(self):
        """."""
        return self.backend_e.voltage

    @voltage.setter
    def voltage(self, value):
        """."""
        self.backend_e.voltage = value

    @property
    def phase_lag(self):
        """."""
        return self.backend_e.phase_lag

    @phase_lag.setter
    def phase_lag(self, value):
        """."""
        self.backend_e.phase_lag = value

    # @property
    # def kicktable(self):
    #     """."""
    #     if self.backend_e.kicktable_idx != -1:
    #         kicktable = _trackcpp.cvar.kicktable_list[self.backend_e.kicktable_idx]
    #         return Kicktable(kicktable=kicktable)
    #     else:
    #         return None

    # @kicktable.setter
    # def kicktable(self, value):
    #     """."""
    #     if not isinstance(value, Kicktable):
    #         raise TypeError('value must be of Kicktable type')
    #     self.backend_e.kicktable = value._kicktable

    @property
    def rescale_kicks(self):
        """Scale factor applied to values given by kicktable.
        Default value: 1.0"""
        return self.backend_e.rescale_kicks

    @rescale_kicks.setter
    def rescale_kicks(self, value):
        """."""
        self.backend_e.rescale_kicks = value

    @property
    def vchamber(self):
        """Shape of vacuum chamber.
        See trackcpp.VChamberShape for values."""
        return backend.Int(self.backend_e.vchamber)

    @vchamber.setter
    def vchamber(self, value):
        """Set shape of vacuum chamber.
        See trackcpp.VChamberShape for values."""
        if value >= 0:
            self.backend_e.vchamber = backend.VChamberShape(value)
        else:
            raise ValueError('Invalid vchamber p-norm number.')

    @property
    def hmax(self):
        """."""
        return self.backend_e.hmax

    @hmax.setter
    def hmax(self, value):
        """."""
        self.backend_e.hmax = value

    @property
    def hmin(self):
        """."""
        return self.backend_e.hmin

    @hmin.setter
    def hmin(self, value):
        """."""
        self.backend_e.hmin = value

    @property
    def vmax(self):
        """."""
        return self.backend_e.vmax

    @vmax.setter
    def vmax(self, value):
        """."""
        self.backend_e.vmax = value

    @property
    def vmin(self):
        """."""
        return self.backend_e.vmin

    @vmin.setter
    def vmin(self, value):
        """."""
        self.backend_e.vmin = value

    @property
    def K(self):
        """."""
        return self.backend_e.polynom_b[1+backend.I_SHIFT]

    @K.setter
    def K(self, value):
        """."""
        self.backend_e.polynom_b[1+backend.I_SHIFT] = value

    @property
    def KL(self):
        """."""
        return self.backend_e.polynom_b[1+backend.I_SHIFT] * self.backend_e.length

    @KL.setter
    def KL(self, value):
        """."""
        self.backend_e.polynom_b[1+backend.I_SHIFT] = value / self.backend_e.length

    @property
    def KxL(self):
        """."""
        return -self.backend_e.matrix66[1][0]

    @KxL.setter
    def KxL(self, value):
        """."""
        lst = list(self.backend_e.matrix66[1])
        lst[0] = -value
        self.backend_e.matrix66[1] = tuple(lst)

    @property
    def KyL(self):
        """."""
        return -self.backend_e.matrix66[3][2]

    @KyL.setter
    def KyL(self, value):
        """."""
        lst = list(self.backend_e.matrix66[3])
        lst[2] = -value
        self.backend_e.matrix66[3] = tuple(lst)

    @property
    def S(self):
        """."""
        return self.backend_e.polynom_b[2+backend.I_SHIFT]

    @S.setter
    def S(self, value):
        """."""
        self.backend_e.polynom_b[2+backend.I_SHIFT] = value

    @property
    def SL(self):
        """."""
        return self.backend_e.polynom_b[2+backend.I_SHIFT] * self.backend_e.length

    @SL.setter
    def SL(self, value):
        self.backend_e.polynom_b[2+backend.I_SHIFT] = value / self.backend_e.length

    @property
    def Ks(self):
        """."""
        return -self.backend_e.polynom_a[1+backend.I_SHIFT]

    @Ks.setter
    def Ks(self, value):
        """."""
        self.backend_e.polynom_a[1+backend.I_SHIFT] = -value

    @property
    def KsL(self):
        """."""
        return -self.backend_e.polynom_a[1+backend.I_SHIFT] * self.backend_e.length

    @KsL.setter
    def KsL(self, value):
        """."""
        self.backend_e.polynom_a[1+backend.I_SHIFT] = -value / self.backend_e.length

    @property
    def KsxL(self):
        """."""
        return -self.backend_e.matrix66[1][2]

    @KsxL.setter
    def KsxL(self, value):
        """."""
        lst = list(self.backend_e.matrix66[1])
        lst[2] = -value
        self.backend_e.matrix66[1] = tuple(lst)

    @property
    def KsyL(self):
        """."""
        return -self.backend_e.matrix66[3][0]

    @KsyL.setter
    def KsyL(self, value):
        """."""
        lst = list(self.backend_e.matrix66[3])
        lst[0] = -value
        self.backend_e.matrix66[3] = tuple(lst)

    @property
    def hkick_polynom(self):
        """."""
        return self.backend_e.polynom_b[0+backend.I_SHIFT] * (-self.backend_e.length)

    @hkick_polynom.setter
    def hkick_polynom(self, value):
        """."""
        self.backend_e.polynom_b[0+backend.I_SHIFT] = - value / self.backend_e.length

    @property
    def vkick_polynom(self):
        """."""
        return self.backend_e.polynom_a[0+backend.I_SHIFT] * self.backend_e.length

    @vkick_polynom.setter
    def vkick_polynom(self, value):
        """."""
        self.backend_e.polynom_a[0+backend.I_SHIFT] = value / self.backend_e.length

    @property
    def polynom_a(self):
        """."""
        return _Polynom(self.backend_e.polynom_a)

    @polynom_a.setter
    def polynom_a(self, value):
        """."""
        if self._backend_type == "cpp":
            self.backend_e.polynom_a[:] = value[:]
        elif self._backend_type == "julia":
            self.backend_e.polynom_a = value
        else:
            ValueError("invalid backend")

    @property
    def polynom_b(self):
        """."""
        return _Polynom(self.backend_e.polynom_b)

    @polynom_b.setter
    def polynom_b(self, value):
        """."""
        if self._backend_type == "cpp":
            self.backend_e.polynom_b[:] = value[:]
        elif self._backend_type == "julia":
            self.backend_e.polynom_b = value
        else:
            ValueError("invalid backend")

    @property
    def matrix66(self):
        """."""
        return _numpy.array(self.backend_e.matrix66)

    @matrix66.setter
    def matrix66(self, value):
        """."""
        tups = []
        for i in range(6):
            tups.append(tuple(float(value[i][j]) for j in range(6)))
        for i in range(6):
            self.backend_e.matrix66[i] = tups[i]

    @property
    def t_in(self):
        """."""
        return backend.get_array(self.backend_e.t_in)

    @t_in.setter
    def t_in(self, value):
        """."""
        Element._check_type(value, Element._t_valid_types)
        Element._check_size(value, backend._NUM_COORDS)
        backend.set_array_from_vector(
            self.backend_e.t_in, backend._NUM_COORDS, value)

    @property
    def t_out(self):
        """."""
        return backend.get_array(self.backend_e.t_out)

    @t_in.setter
    def t_out(self, value):
        """."""
        Element._check_type(value, Element._t_valid_types)
        Element._check_size(value, backend._NUM_COORDS)
        backend.set_array_from_vector(
            self.backend_e.t_out, backend._NUM_COORDS, value)

    @property
    def r_in(self):
        """."""
        return backend.get_matrix(self.backend_e.r_in)

    @r_in.setter
    def r_in(self, value):
        """."""
        Element._check_type(value, Element._r_valid_types)
        Element._check_shape(value, backend._DIMS)
        backend.set_array_from_matrix(self.backend_e.r_in, backend._DIMS, value)

    @property
    def r_out(self):
        """."""
        return backend.get_matrix(self.backend_e.r_out)

    @r_in.setter
    def r_out(self, value):
        """."""
        Element._check_type(value, Element._r_valid_types)
        Element._check_shape(value, backend._DIMS)
        backend.set_array_from_matrix(self.backend_e.r_out, backend._DIMS, value)

    def __eq__(self, other):
        """."""
        if not isinstance(other, Element):
            return NotImplemented
        if self._backend_type != other._backend_type:
            print("Different backends")
            return False
        for attr in _ELEM_ATTRIBUTES:
            self_attr = getattr(self, attr)
            if isinstance(self_attr, _numpy.ndarray):
                if (self_attr != getattr(other, attr)).any():
                    return False
            else:
                if self_attr != getattr(other, attr):
                    return False
        return True

    def __repr__(self):
        """."""
        return 'fam_name: ' + self.fam_name

    def __str__(self):
        """."""
        fmtstr = '\n{0:<11s}: {1} {2}'
        rst = ''
        rst += fmtstr[1:].format('fam_name', self.fam_name, '')
        rst += fmtstr.format('pass_method', self.pass_method, '')
        if self.length != 0:
            rst += fmtstr.format('length', self.length, 'm')
        if self.nr_steps != 1:
            rst += fmtstr.format('nr_steps', self.nr_steps, '')
        if self.angle != 0:
            rst += fmtstr.format('angle', self.angle, 'rad')
        if self.angle_in != 0:
            rst += fmtstr.format('angle_in', self.angle_in, 'rad')
        if self.angle_out != 0:
            rst += fmtstr.format('angle_out', self.angle_out, 'rad')
        if self.gap != 0:
            rst += fmtstr.format('gap', self.gap, 'm')
        if self.fint_in != 0:
            rst += fmtstr.format('fint_in', self.fint_in, '')
        if self.fint_out != 0:
            rst += fmtstr.format('fint_out', self.fint_out, '')

        if self.thin_KL != 0:
            rst += fmtstr.format('thin_KL', self.thin_KL, '1/m')
        if self.thin_SL != 0:
            rst += fmtstr.format('thin_SL', self.thin_SL, '1/m²')

        if not all([v == 0 for v in self.polynom_a]):
            rst += fmtstr.format(
                'polynom_a', self.polynom_a, '1/m¹, 1/m², 1/m³, ...')
        if not all([v == 0 for v in self.polynom_b]):
            rst += fmtstr.format(
                'polynom_b', self.polynom_b, '1/m¹, 1/m², 1/m³, ...')
        if self.hkick != 0:
            rst += fmtstr.format('hkick', self.hkick, 'rad')
        if self.vkick != 0:
            rst += fmtstr.format('vkick', self.vkick, 'rad')
        if self.frequency != 0:
            rst += fmtstr.format('frequency', self.frequency, 'Hz')
        if self.voltage != 0:
            rst += fmtstr.format('voltage', self.voltage, 'V')
        if self.phase_lag != 0:
            rst += fmtstr.format('phase_lag', self.phase_lag, 'rad')
        if self.vchamber != 0:
            rst += fmtstr.format('vchamber', self.vchamber, '')

        if self.hmin != -backend.float_MAX:
            rst += fmtstr.format('hmin', self.hmin, 'm')
        if self.hmax != backend.float_MAX:
            rst += fmtstr.format('hmax', self.hmax, 'm')
        if self.vmin != -backend.float_MAX:
            rst += fmtstr.format('vmin', self.vmin, 'm')
        if self.vmax != backend.float_MAX:
            rst += fmtstr.format('vmax', self.vmax, 'm')

        # if self.backend_e.kicktable_idx != -1:
        #     kicktable = _trackcpp.cvar.kicktable_list[self.backend_e.kicktable_idx]
        #     rst += fmtstr.format('kicktable', kicktable.filename, '')

        if not (self.t_in == _numpy.zeros(backend._NUM_COORDS)).all():
            rst += fmtstr.format('t_in', self.t_in, 'm')
        if not (self.t_out == _numpy.zeros(backend._NUM_COORDS)).all():
            rst += fmtstr.format('t_out', self.t_out, 'm')
        if not (self.r_in == _numpy.eye(backend._NUM_COORDS)).all():
            rst += fmtstr.format('r_in', '6x6 matrix', '')
        if not (self.r_out == _numpy.eye(backend._NUM_COORDS)).all():
            rst += fmtstr.format('r_out', '6x6 matrix', '')

        if not (self.matrix66 == _numpy.eye(backend._NUM_COORDS)).all():
            rst += fmtstr.format('matrix66', '6x6 matrix', '')

        return rst

    # --- private methods ---

    @staticmethod
    def _check_type(value, types):
        """."""
        res = False
        for typ in types:
            res = res or isinstance(value, typ)
        if not res:
            raise TypeError("value must be list or numpy.ndarray")

    @staticmethod
    def _check_size(value, size):
        if not len(value) == size:
            raise ValueError("size must be " + str(size))

    @staticmethod
    def _check_shape(value, shape):
        if not value.shape == shape:
            raise ValueError("shape must be " + str(shape))


_warnings.filterwarnings(
    "ignore", "Item size computed from the PEP 3118 \
    buffer format string does not match the actual item size.")
