
# -*- coding: utf-8 -*-

import ctypes as _ctypes
import warnings as _warnings
import numpy as _numpy
import trackcpp as _trackcpp
from pyaccel.utils import interactive as _interactive
from pyaccel.utils import Polynom as _Polynom


_DBL_MAX = _trackcpp.get_double_max()
_NUM_COORDS = 6
_DIMS = (_NUM_COORDS, _NUM_COORDS)
_coord_vector = _ctypes.c_double*_NUM_COORDS
_coord_matrix = _ctypes.c_double*_DIMS[0]*_DIMS[1]

pass_methods = _trackcpp.pm_dict


@_interactive
def marker(fam_name):
    """Create a marker element.

    Keyword arguments:
    fam_name -- family name
    """
    e = _trackcpp.marker_wrapper(fam_name)
    return Element(element=e)


@_interactive
def bpm(fam_name):
    """Create a beam position monitor element.

    Keyword arguments:
    fam_name -- family name
    """
    e = _trackcpp.bpm_wrapper(fam_name)
    return Element(element=e)


@_interactive
def drift(fam_name, length):
    """Create a drift element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    """
    e = _trackcpp.drift_wrapper(fam_name, length)
    return Element(element=e)


@_interactive
def hcorrector(fam_name,  length=0.0, hkick=0.0):
    """Create a horizontal corrector element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    hkick -- horizontal kick [rad]
    """
    e = _trackcpp.hcorrector_wrapper(fam_name, length, hkick)
    return Element(element=e)


@_interactive
def vcorrector(fam_name, length=0.0, vkick=0.0):
    """Create a vertical corrector element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    vkick -- vertical kick [rad]
    """
    e = _trackcpp.vcorrector_wrapper(fam_name, length, vkick)
    return Element(element=e)


@_interactive
def corrector(fam_name,  length=0.0, hkick=0.0, vkick=0.0):
    """Create a corrector element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    hkick -- horizontal kick [rad]
    vkick -- vertical kick [rad]
    """
    e = _trackcpp.corrector_wrapper(fam_name, length, hkick, vkick)
    return Element(element=e)


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
    if K is None: K = polynom_b[1]
    if S is None: S = polynom_b[2]
    e = _trackcpp.rbend_wrapper(fam_name, length, angle, angle_in,
            angle_out, gap, fint_in, fint_out, polynom_a, polynom_b,
            K, S)
    return Element(element=e)


@_interactive
def quadrupole(fam_name, length, K, nr_steps=10):
    """Create a quadrupole element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    K -- [m^-2]
    nr_steps -- number of steps (default 10)
    """
    e = _trackcpp.quadrupole_wrapper(fam_name, length, K, nr_steps)
    return Element(element=e)


@_interactive
def sextupole(fam_name, length, S, nr_steps=5):
    """Create a sextupole element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    S -- (1/2!)(d^2By/dx^2)/(Brho)[m^-3]
    nr_steps -- number of steps (default 5)
    """
    e = _trackcpp.sextupole_wrapper(fam_name, length, S, nr_steps)
    return Element(element=e)


@_interactive
def rfcavity(fam_name, length, voltage, frequency):
    """Create a RF cavity element.

    Keyword arguments:
    fam_name -- family name
    length -- [m]
    voltage -- [V]
    frequency -- [Hz]
    """
    e = _trackcpp.rfcavity_wrapper(fam_name, length, frequency, voltage)
    return Element(element=e)


def _process_polynoms(pa, pb):
    # Make sure pa and pb have same size and are initialized
    if pa is None:
        pa = [0.0,0.0,0.0]
    if pb is None:
        pb = [0.0,0.0,0.0]
    n = max([3, len(pa), len(pb)])
    for i in range(len(pa), n):
        pa.append(0.0)
    for i in range(len(pb), n):
        pb.append(0.0)
    return pa, pb


@_interactive
class Kicktable(object):

    def __init__(self, **kwargs):
        if 'kicktable' in kwargs:
            self._kicktable = kwargs['kicktable']
        else:
            filename = kwargs.get('filename', "")
            self._kicktable = _trackcpp.Kicktable(filename)

    def __eq__(self,other):
        if not isinstance(other,Kicktable): return NotImplemented
        for attr in self._kicktable.__swig_getmethods__:
            if getattr(self,attr) != getattr(other,attr):
                return False
        return True

    @property
    def filename(self):
        return self._kicktable.filename

    @property
    def length(self):
        return self._kicktable.length

    @property
    def x_min(self):
        return self._kicktable.x_min

    @property
    def x_max(self):
        return self._kicktable.x_max

    @property
    def y_min(self):
        return self._kicktable.y_min

    @property
    def y_max(self):
        return self._kicktable.y_max

    @property
    def x_nrpts(self):
        return self._kicktable.x_nrpts

    @property
    def y_nrpts(self):
        return self._kicktable.y_nrpts


class Element(object):

    _t_valid_types = (list, _numpy.ndarray)
    _r_valid_types = (_numpy.ndarray, )

    def __init__(self, **kwargs):
        if 'element' in kwargs:
            if isinstance(kwargs['element'],_trackcpp.Element):
                copy = kwargs.get('copy',False)
                if copy:
                    self._e = _trackcpp.Element(kwargs['element'])
                else:
                    self._e = kwargs['element']
            elif isinstance(kwargs['element'],Element):
                copy = kwargs.get('copy',True)
                if copy:
                    self._e = _trackcpp.Element(kwargs['element']._e)
                else:
                    self._e = kwargs['element']._e
            else:
                raise TypeError('element must be a trackcpp.Element or a Element object.')
        else:
            fam_name = kwargs.get('fam_name', "")
            length = kwargs.get('length', 0.0)
            self._e = _trackcpp.Element(fam_name, length)

    def __eq__(self,other):
        if not isinstance(other,Element): return NotImplemented
        for attr in self._e.__swig_getmethods__:
            self_attr = getattr(self,attr)
            if isinstance(self_attr,_numpy.ndarray):
                if (self_attr != getattr(other,attr)).any():
                    return False
            else:
                if self_attr != getattr(other,attr):
                    return False
        return True


    @property
    def fam_name(self):
        return self._e.fam_name

    @fam_name.setter
    def fam_name(self, value):
        self._e.fam_name = value

    @property
    def pass_method(self):
        return pass_methods[self._e.pass_method]

    @pass_method.setter
    def pass_method(self, value):
        if isinstance(value, str):
            if value not in pass_methods:
                raise ValueError("pass method '" + value + "' not found")
            else:
                self._e.pass_method = pass_methods.index(value)
        elif isinstance(value, int):
            if not (0 <= value < len(pass_methods)):
                raise IndexError("pass method index out of range")
            else:
                self._e.pass_method = value
        else:
            raise TypeError("pass method value must be string or index")

    @property
    def length(self):
        return self._e.length

    @length.setter
    def length(self, value):
        self._e.length = value

    @property
    def nr_steps(self):
        return self._e.nr_steps

    @nr_steps.setter
    def nr_steps(self, value):
        self._e.nr_steps = value

    @property
    def hkick(self):
        return self._e.hkick

    @hkick.setter
    def hkick(self, value):
        self._e.hkick = value

    @property
    def vkick(self):
        return self._e.vkick

    @vkick.setter
    def vkick(self, value):
        self._e.vkick = value

    @property
    def angle(self):
        return self._e.angle

    @angle.setter
    def angle(self, value):
        self._e.angle = value

    @property
    def angle_in(self):
        return self._e.angle_in

    @angle_in.setter
    def angle_in(self, value):
        self._e.angle_in = value

    @property
    def angle_out(self):
        return self._e.angle_out

    @angle_out.setter
    def angle_out(self, value):
        self._e.angle_out = value

    @property
    def gap(self):
        return self._e.gap

    @gap.setter
    def gap(self, value):
        self._e.gap = value

    @property
    def fint_in(self):
        return self._e.fint_in

    @fint_in.setter
    def fint_in(self, value):
        self._e.fint_in = value

    @property
    def fint_out(self):
        return self._e.fint_out

    @fint_out.setter
    def fint_out(self, value):
        self._e.fint_out = value

    @property
    def thin_KL(self):
        return self._e.thin_KL

    @thin_KL.setter
    def thin_KL(self, value):
        self._e.thin_KL = value

    @property
    def thin_SL(self):
        return self._e.thin_SL

    @thin_SL.setter
    def thin_SL(self, value):
        self._e.thin_SL = value

    @property
    def frequency(self):
        return self._e.frequency

    @frequency.setter
    def frequency(self, value):
        self._e.frequency = value

    @property
    def voltage(self):
        return self._e.voltage

    @voltage.setter
    def voltage(self, value):
        self._e.voltage = value

    @property
    def kicktable(self):
        if self._e.kicktable is not None:
            return Kicktable(kicktable=self._e.kicktable)
        else:
            return None

    @kicktable.setter
    def kicktable(self, value):
        if not isinstance(value, Kicktable):
            raise TypeError('value must be of Kicktable type')
        self._e.kicktable = value._kicktable

    @property
    def hmax(self):
        return self._e.hmax

    @hmax.setter
    def hmax(self, value):
        self._e.hmax = value

    @property
    def hmin(self):
        return self._e.hmin

    @hmin.setter
    def hmin(self, value):
        self._e.hmin = value

    @property
    def vmax(self):
        return self._e.vmax

    @vmax.setter
    def vmax(self, value):
        self._e.vmax = value

    @property
    def vmin(self):
        return self._e.vmin

    @vmin.setter
    def vmin(self, value):
        self._e.vmin = value

    @property
    def K(self):
        return self._e.polynom_b[1]

    @K.setter
    def K(self, value):
        self._e.polynom_b[1] = value

    @property
    def S(self):
        return self._e.polynom_b[2]

    @S.setter
    def S(self, value):
        self._e.polynom_b[2] = value

    @property
    def Ks(self):
        return self._e.polynom_a[1]

    @Ks.setter
    def Ks(self, value):
        self._e.polynom_a[1] = value

    @property
    def hkick_polynom(self):
        return self._e.polynom_b[0] *(- self._e.length)

    @hkick_polynom.setter
    def hkick_polynom(self, value):
        self._e.polynom_b[0] = - value / self._e.length

    @property
    def vkick_polynom(self):
        return self._e.polynom_a[0] * self._e.length

    @vkick_polynom.setter
    def vkick_polynom(self, value):
        self._e.polynom_a[0] = value / self._e.length

    @property
    def polynom_a(self):
        p = _Polynom(self._e.polynom_a)
        return p

    @polynom_a.setter
    def polynom_a(self, value):
        self._e.polynom_a[:] = value[:]

    @property
    def polynom_b(self):
        p = _Polynom(self._e.polynom_b)
        return p

    @polynom_b.setter
    def polynom_b(self, value):
        self._e.polynom_b[:] = value[:]

    @property
    def t_in(self):
        return self._get_coord_vector(self._e.t_in)

    @t_in.setter
    def t_in(self, value):
        self._check_type(value, Element._t_valid_types)
        self._check_size(value, _NUM_COORDS)
        self._set_c_array_from_vector(self._e.t_in, _NUM_COORDS, value)

    @property
    def t_out(self):
        return self._get_coord_vector(self._e.t_out)

    @t_out.setter
    def t_out(self, value):
        self._check_type(value, Element._t_valid_types)
        self._check_size(value, _NUM_COORDS)
        self._set_c_array_from_vector(self._e.t_out, _NUM_COORDS, value)

    @property
    def r_in(self):
        return self._get_coord_matrix(self._e.r_in)

    @r_in.setter
    def r_in(self, value):
        self._check_type(value, Element._r_valid_types)
        self._check_shape(value, _DIMS)
        self._set_c_array_from_matrix(self._e.r_in, _DIMS, value)

    @property
    def r_out(self):
        return self._get_coord_matrix(self._e.r_out)

    @r_out.setter
    def r_out(self, value):
        self._check_type(value, Element._r_valid_types)
        self._check_shape(value, _DIMS)
        self._set_c_array_from_matrix(self._e.r_out, _DIMS, value)

    def _set_c_array_from_vector(self, array, size, values):
        if not (size == len(values)):
            raise ValueError("array and vector must have same size")
        for i in range(size):
            _trackcpp.c_array_set(array, i, values[i])

    def _set_c_array_from_matrix(self, array, shape, values):
        if not (shape == values.shape):
            raise ValueError("array and matrix must have same shape")
        rows, cols = shape
        for i in range(rows):
            for j in range(cols):
                _trackcpp.c_array_set(array, i*cols + j, values[i, j])

    def _check_type(self, value, types):
        r = False
        for t in types:
            r = r or isinstance(value, t)
        if not r:
            raise TypeError("value must be list or numpy.ndarray")

    def _check_size(self, value, size):
        if not len(value) == size:
            raise ValueError("size must be " + str(size))

    def _check_shape(self, value, shape):
        if not value.shape == shape:
            raise ValueError("shape must be " + str(shape))

    def _get_coord_vector(self, pointer):
        address = int(pointer)
        c_array = _coord_vector.from_address(address)
        return _numpy.ctypeslib.as_array(c_array)

    def _get_coord_matrix(self, pointer):
        address = int(pointer)
        c_array = _coord_matrix.from_address(address)
        return _numpy.ctypeslib.as_array(c_array)

    def __repr__(self):
        return 'fam_name: ' + self.fam_name

    def __str__(self):
        fmtstr = '\n{0:<11s}: {1} {2}'
        r  =   ''
        r += fmtstr[1:].format('fam_name', self.fam_name, '')
        r += fmtstr.format('pass_method', self.pass_method, '')
        if self.length != 0:
            r += fmtstr.format('length', self.length, 'm')
        if self.nr_steps != 1:
            r += fmtstr.format('nr_steps', self.nr_steps, '')
        if self.angle != 0:
            r += fmtstr.format('angle', self.angle, 'rad')
        if self.angle_in != 0:
            r += fmtstr.format('angle_in', self.angle_in, 'rad')
        if self.angle_out != 0:
            r += fmtstr.format('angle_out', self.angle_out, 'rad')
        if self.gap != 0:
            r += fmtstr.format('gap', self.gap, 'm')
        if self.fint_in != 0:
            r += fmtstr.format('fint_in', self.fint_in, '')
        if self.fint_out != 0:
            r += fmtstr.format('fint_out', self.fint_out, '')
        if self.thin_KL != 0:
            r += fmtstr.format('thin_KL', self.thin_KL, '1/m')
        if self.thin_SL != 0:
            r += fmtstr.format('thin_SL', self.thin_SL, '1/m²')
        if not all([v == 0 for v in self.polynom_a]):
            r += fmtstr.format('polynom_a', self.polynom_a, '1/m¹, 1/m², 1/m³, ...')
        if not all([v == 0 for v in self.polynom_b]):
            r += fmtstr.format('polynom_b', self.polynom_b, '1/m¹, 1/m², 1/m³, ...')
        if self.hkick != 0:
            r += fmtstr.format('hkick', self.hkick, 'rad')
        if self.vkick != 0:
            r += fmtstr.format('vkick', self.vkick, 'rad')
        if self.frequency != 0:
            r += fmtstr.format('frequency', self.frequency, 'Hz')
        if self.voltage != 0:
            r += fmtstr.format('voltage', self.voltage, 'V')
        if self.hmin < _DBL_MAX:
            r += fmtstr.format('hmin', self.hmin, 'm')
        if self.hmax < _DBL_MAX:
            r += fmtstr.format('hmax', self.hmax, 'm')
        if self.vmin < _DBL_MAX:
            r += fmtstr.format('vmin', self.vmin, 'm')
        if self.vmax < _DBL_MAX:
            r += fmtstr.format('vmax', self.vmax, 'm')
        if self.kicktable is not None:
            r += fmtstr.format('kicktable', self.kicktable.filename, '')
        if not (self.t_in == _numpy.zeros(_NUM_COORDS)).all():
            r += fmtstr.format('t_in', self.t_in, 'm')
        if not (self.t_out == _numpy.zeros(_NUM_COORDS)).all():
            r += fmtstr.format('t_out', self.t_out, 'm')
        if not (self.r_in == _numpy.eye(_NUM_COORDS)).all():
            r += fmtstr.format('r_in', '6x6 matrix', '')
        if not (self.r_out == _numpy.eye(_NUM_COORDS)).all():
            r += fmtstr.format('r_out', '6x6 matrix', '')

        return r


_warnings.filterwarnings("ignore", "Item size computed from the PEP 3118 buffer format string does not match the actual item size.")
