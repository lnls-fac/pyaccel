"""Accelerator class."""

import mathphys as _mp
import numpy as _np
import trackcpp as _trackcpp
from mathphys.functions import get_namedtuple as _get_namedtuple

from . import elements as _elements
from .utils import interactive as _interactive


class AcceleratorError(Exception):
    """."""


@_interactive
class Accelerator(object):
    """."""

    _states_str = tuple(state.title() for state in _trackcpp.rad_dict)
    RadiationStates = _get_namedtuple('RadiationStates', _states_str)
    del _states_str

    __isfrozen = False  # this is used to prevent creation of new attributes

    def __init__(self, **kwargs):
        """."""
        self.trackcpp_acc = self._init_accelerator(kwargs)
        self._init_lattice(kwargs)

        if 'energy' in kwargs:
            self.trackcpp_acc.energy = kwargs['energy']
        if 'harmonic_number' in kwargs:
            self.trackcpp_acc.harmonic_number = kwargs['harmonic_number']
        if 'radiation_on' in kwargs:
            self.radiation_on = kwargs['radiation_on']
        if 'cavity_on' in kwargs:
            self.trackcpp_acc.cavity_on = kwargs['cavity_on']
        if 'vchamber_on' in kwargs:
            self.trackcpp_acc.vchamber_on = kwargs['vchamber_on']
        if 'lattice_version' in kwargs:
            self.trackcpp_acc.lattice_version = kwargs['lattice_version']

        if self.trackcpp_acc.energy == 0:
            self._brho, self._velocity, self._beta, self._gamma, \
                self.trackcpp_acc.energy = \
                _mp.beam_optics.beam_rigidity(gamma=1.0)
        else:
            self._brho, self._velocity, self._beta, self._gamma, energy = \
                _mp.beam_optics.beam_rigidity(energy=self.energy/1e9)
            self.trackcpp_acc.energy = energy * 1e9

        self.__isfrozen = True

    @property
    def length(self):
        """Return lattice length [m]."""
        return self.trackcpp_acc.get_length()

    @property
    def energy(self):
        """Return beam energy [eV]."""
        return self.trackcpp_acc.energy

    @energy.setter
    def energy(self, value):
        """."""
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(energy=value/1e9)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def gamma_factor(self):
        """Return beam relativistic gamma factor."""
        return self._gamma

    @gamma_factor.setter
    def gamma_factor(self, value):
        """Set beam relativistic gamma factor."""
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(gamma=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def beta_factor(self):
        """Return beam relativistic beta factor."""
        return self._beta

    @beta_factor.setter
    def beta_factor(self, value):
        """Set beam relativistic beta factor."""
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(beta=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def velocity(self):
        """Return beam velocity [m/s]."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        """Set beam velocity [m/s]."""
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(velocity=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def brho(self):
        """Return beam rigidity [T.m]."""
        return self._brho

    @brho.setter
    def brho(self, value):
        """Set beam rigidity [T.m]."""
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(brho=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def harmonic_number(self):
        """Return accelerator harmonic number."""
        return self.trackcpp_acc.harmonic_number

    @harmonic_number.setter
    def harmonic_number(self, value):
        """Set accelerator harmonic number."""
        if not isinstance(value, int) or value < 1:
            raise AcceleratorError(
                'harmonic number has to be a positive integer')
        self.trackcpp_acc.harmonic_number = value

    @property
    def cavity_on(self):
        """Return cavity on state."""
        return self.trackcpp_acc.cavity_on

    @cavity_on.setter
    def cavity_on(self, value):
        """Set cavity on state."""
        if self.trackcpp_acc.harmonic_number < 1:
            raise AcceleratorError('invalid harmonic number')
        self.trackcpp_acc.cavity_on = value

    @property
    def radiation_on(self):
        """Return radiation on state."""
        return self.trackcpp_acc.radiation_on

    @property
    def radiation_on_str(self):
        """Return radiation_on state in string format."""
        return self.RadiationStates._fields[self.trackcpp_acc.radiation_on]

    @radiation_on.setter
    def radiation_on(self, value):
        """Set radiation on state.

        Args:
            value (int, bool or string): Radiation state to be set, the
                options are:
                    - 0, False, "Off"    = No radiative effects.
                    - 1, True, "Damping" = Turns on radiation damping, without
                        quantum excitation.
                    - 2, "Full" = Turns on radiation damping with quantum
                        excitation

        Raises:
            ValueError: for wrong values in `value` input.
        """
        radstts = self.RadiationStates
        if isinstance(value, (int, bool, float)) and int(value) in radstts:
            self.trackcpp_acc.radiation_on = int(value)
        elif isinstance(value, str) and value.title() in radstts._fields:
            self.trackcpp_acc.radiation_on = \
                radstts._fields.index(value.title())
        else:
            raise ValueError('Value not valid.')

    @property
    def vchamber_on(self):
        """Return vacuum chamber on state."""
        return self.trackcpp_acc.vchamber_on

    @vchamber_on.setter
    def vchamber_on(self, value):
        """Set vacuum chamber on state."""
        self.trackcpp_acc.vchamber_on = value

    @property
    def lattice_version(self):
        """Return lattice version."""
        return self.trackcpp_acc.lattice_version

    @lattice_version.setter
    def lattice_version(self, value):
        """Set lattice version."""
        self.trackcpp_acc.lattice_version = value

    def pop(self, index):
        """."""
        elem = self[index]
        del self[index]
        return elem

    def append(self, element: _elements.Element):
        """Append element to end of lattice.

        Args:
            element (pyaccel.elements.Element): Element to added.

        Raises:
            TypeError: when element is not an pyaccel.elements.Element.

        """
        if not isinstance(element, _elements.Element):
            raise TypeError('value must be Element')
        self.trackcpp_acc.lattice.append(element.trackcpp_e)

    def insert(self, index: int, element: _elements.Element):
        """Insert element at a given index of the lattice.

        Args:
            index (int): index where to insert element. May be negative or
                positive
            element (pyaccel.elements.Element): Element to be inserted.

        Raises:
            TypeError: when element is not a pyaccel.elements.Element.

        """
        if not isinstance(element, _elements.Element):
            raise TypeError('element must be of type pyaccel.elements.Element')

        leng = len(self)
        index = int(index)
        index = max(min(index, leng), -leng)
        if index < 0:
            index += leng
        idx = self.trackcpp_acc.lattice.begin() + index
        self.trackcpp_acc.lattice.insert(idx, element.trackcpp_e)

    def extend(self, value):
        """."""
        if not isinstance(value, Accelerator):
            raise TypeError('value must be Accelerator')
        if value is self:
            value = Accelerator(accelerator=value)
        for el in value:
            self.append(el)

    # NOTE: make the class objects pickalable
    def __getstate__(self):
        """."""
        stri = _trackcpp.String()
        _trackcpp.write_flat_file_wrapper(stri, self.trackcpp_acc, False)
        return stri.data

    def __setstate__(self, stridata):
        """."""
        stri = _trackcpp.String(stridata)
        acc = Accelerator()
        _trackcpp.read_flat_file_wrapper(stri, acc.trackcpp_acc, False)
        self.trackcpp_acc = acc.trackcpp_acc

    def __setattr__(self, key, value):
        """."""
        if self.__isfrozen and not hasattr(self, key):
            raise AcceleratorError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def __delitem__(self, index):
        """."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            index = set(range(start, stop, step))
        if isinstance(index, (int, _np.int_)):
            self.trackcpp_acc.lattice.erase(
                self.trackcpp_acc.lattice.begin() + int(index))
        elif isinstance(index, (set, list, tuple, _np.ndarray)):
            index = sorted(set(index), reverse=True)
            for i in index:
                self.trackcpp_acc.lattice.erase(
                    self.trackcpp_acc.lattice.begin() + int(i))

    def __getitem__(self, index):
        """."""
        if isinstance(index, (int, _np.int_)):
            ele = _elements.Element()
            ele.trackcpp_e = self.trackcpp_acc.lattice[int(index)]
            return ele
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self))))

        if isinstance(index, (list, tuple, _np.ndarray)):
            try:
                index = _np.array(index, dtype=int)
            except TypeError:
                raise TypeError('invalid index')
            lattice = _trackcpp.CppElementVector()
            for i in index:
                lattice.append(self.trackcpp_acc.lattice[int(i)])
        else:
            raise TypeError('invalid index')
        acc = Accelerator(
            lattice=lattice,
            lattice_version=self.trackcpp_acc.lattice_version,
            energy=self.trackcpp_acc.energy,
            harmonic_number=self.trackcpp_acc.harmonic_number,
            cavity_on=self.trackcpp_acc.cavity_on,
            radiation_on=self.trackcpp_acc.radiation_on,
            vchamber_on=self.trackcpp_acc.vchamber_on)
        return acc

    def __setitem__(self, index, value):
        """."""
        if isinstance(index, (int, _np.int_)):
            index = [index, ]
        elif isinstance(index, (list, tuple, _np.ndarray)):
            pass
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            index = range(start, stop, step)
        else:
            raise TypeError('invalid index')

        if isinstance(value, (list, tuple, _np.ndarray, Accelerator)):
            if not all([isinstance(v, _elements.Element) for v in value]):
                raise TypeError('invalid value')
            for i, val in zip(index, value):
                self.trackcpp_acc.lattice[int(i)] = val.trackcpp_e
        elif isinstance(value, _elements.Element):
            for i in index:
                self.trackcpp_acc.lattice[int(i)] = value.trackcpp_e
        else:
            raise TypeError('invalid value')

    def __len__(self):
        """."""
        return self.trackcpp_acc.lattice.size()

    def __str__(self):
        """."""
        rst = ''
        rst += 'energy         : ' + str(self.trackcpp_acc.energy) + ' eV'
        rst += '\nharmonic_number: ' + str(self.trackcpp_acc.harmonic_number)
        rst += '\ncavity_on      : ' + str(self.trackcpp_acc.cavity_on)
        rst += '\nradiation_on   : ' + str(self.trackcpp_acc.radiation_on)
        rst += '\nvchamber_on    : ' + str(self.trackcpp_acc.vchamber_on)
        rst += '\nlattice version: ' + self.trackcpp_acc.lattice_version
        rst += '\nlattice size   : ' + str(len(self.trackcpp_acc.lattice))
        rst += '\nlattice length : ' + str(self.length) + ' m'
        return rst

    def __add__(self, other):
        """."""
        if isinstance(other, _elements.Element):
            acc = self[:]
            acc.append(other)
            return acc
        elif isinstance(other, Accelerator):
            acc = self[:]
            for elem in other:
                acc.append(elem)
            return acc
        else:
            msg = "unsupported operand type(s) for +: '" + \
                    self.__class__.__name__ + "' and '" + \
                    other.__class__.__name__ + "'"
            raise TypeError(msg)

    def __radd__(self, other):
        """."""
        if isinstance(other, _elements.Element):
            acc = self[:]
            acc.insert(0, other)
            return acc
        # if other is of type Accelerator, the __add__ method of other will
        # be called, so we don't need to treat this case here.
        else:
            msg = "unsupported operand type(s) for +: '" + \
                    self.__class__.__name__ + "' and '" + \
                    other.__class__.__name__ + "'"
            raise TypeError(msg)

    def __mul__(self, other):
        """."""
        if isinstance(other, (int, _np.int_)):
            if other < 0:
                raise ValueError('cannot multiply by negative integer')
            elif other == 0:
                return Accelerator(
                    energy=self.energy,
                    lattice_version=self.lattice_version,
                    harmonic_number=self.harmonic_number,
                    cavity_on=self.cavity_on,
                    radiation_on=self.radiation_on,
                    vchamber_on=self.vchamber_on)
            else:
                acc = self[:]
                other -= 1
                while other > 0:
                    acc += self[:]
                    other -= 1
                return acc
        else:
            msg = "unsupported operand type(s) for +: '" + \
                    other.__class__.__name__ + "' and '" + \
                    self.__class__.__name__ + "'"
            raise TypeError(msg)

    def __rmul__(self, other):
        """."""
        return self.__mul__(other)

    def __eq__(self, other):
        """."""
        if not isinstance(other, Accelerator):
            return NotImplemented
        return self.trackcpp_acc.isequal(other.trackcpp_acc)

    # --- private methods ---

    def _init_accelerator(self, kwargs):
        if 'accelerator' in kwargs:
            acc = kwargs['accelerator']
            if isinstance(acc, _trackcpp.Accelerator):
                trackcpp_acc = acc  # points to the same object in memory
            elif isinstance(acc, Accelerator):  # creates another object.
                trackcpp_acc = _trackcpp.Accelerator()
                trackcpp_acc.lattice = acc.trackcpp_acc.lattice[:]
                trackcpp_acc.energy = acc.energy
                trackcpp_acc.cavity_on = acc.cavity_on
                trackcpp_acc.radiation_on = acc.radiation_on
                trackcpp_acc.vchamber_on = acc.vchamber_on
                trackcpp_acc.harmonic_number = acc.harmonic_number
                trackcpp_acc.lattice_version = acc.lattice_version
        else:
            trackcpp_acc = _trackcpp.Accelerator()
            trackcpp_acc.cavity_on = False
            trackcpp_acc.radiation_on = 0  # 0 = radiation off
            trackcpp_acc.vchamber_on = False
            trackcpp_acc.harmonic_number = 0
            trackcpp_acc.lattice_version = ''
        return trackcpp_acc

    def _init_lattice(self, kwargs):
        if 'lattice' in kwargs:
            lattice = kwargs['lattice']
            if isinstance(lattice, _trackcpp.CppElementVector):
                self.trackcpp_acc.lattice = lattice
            elif isinstance(lattice, list):
                for elem in lattice:
                    self.trackcpp_acc.lattice.append(elem.trackcpp_e)
            else:
                raise TypeError('values must be list of Element')
