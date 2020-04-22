
import numpy as _np
import trackcpp as _trackcpp
import pyaccel.lattice as _lattice
import pyaccel.elements as _elements
import mathphys as _mp
from pyaccel.utils import interactive as _interactive


class AcceleratorException(Exception):
    pass


@_interactive
class Accelerator(object):

    __isfrozen = False  # this is used to prevent creation of new attributes

    def __init__(self, **kwargs):

        if 'accelerator' in kwargs:
            a = kwargs['accelerator']
            if isinstance(a, _trackcpp.Accelerator):
                self.trackcpp_acc = a  # points to the same object in memory
            elif isinstance(a, Accelerator):  # creates another object.
                self.trackcpp_acc = _trackcpp.Accelerator()
                self.trackcpp_acc.lattice = a.trackcpp_acc.lattice[:]
                self.trackcpp_acc.energy = a.energy
                self.trackcpp_acc.cavity_on = a.cavity_on
                self.trackcpp_acc.radiation_on = a.radiation_on
                self.trackcpp_acc.vchamber_on = a.vchamber_on
                self.trackcpp_acc.harmonic_number = a.harmonic_number
        else:
            self.trackcpp_acc = _trackcpp.Accelerator()
            self.trackcpp_acc.cavity_on = False
            self.trackcpp_acc.radiation_on = False
            self.trackcpp_acc.vchamber_on = False
            self.trackcpp_acc.harmonic_number = 0

        if 'lattice' in kwargs:
            lattice = kwargs['lattice']
            if isinstance(lattice, _trackcpp.CppElementVector):
                self.trackcpp_acc.lattice = lattice
            elif isinstance(lattice, list):
                for i in range(len(lattice)):
                    e = lattice[i]
                    self.trackcpp_acc.lattice.append(e.trackcpp_e)
            else:
                raise TypeError('values must be list of Element')

        if 'energy' in kwargs:
            self.trackcpp_acc.energy = kwargs['energy']
        if 'harmonic_number' in kwargs:
            self.trackcpp_acc.harmonic_number = kwargs['harmonic_number']
        if 'radiation_on' in kwargs:
            self.trackcpp_acc.radiation_on = kwargs['radiation_on']
        if 'cavity_on' in kwargs:
            self.trackcpp_acc.cavity_on = kwargs['cavity_on']
        if 'vchamber_on' in kwargs:
            self.trackcpp_acc.vchamber_on = kwargs['vchamber_on']

        if self.trackcpp_acc.energy == 0:
            self._brho, self._velocity, self._beta, self._gamma, \
                self.trackcpp_acc.energy = \
                _mp.beam_optics.beam_rigidity(gamma=1.0)
        else:
            self._brho, self._velocity, self._beta, self._gamma, energy = \
                _mp.beam_optics.beam_rigidity(energy=self.energy/1e9)
            self.trackcpp_acc.energy = energy * 1e9

        self.__isfrozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise AcceleratorException("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def __delitem__(self, index):
        if isinstance(index, (int, _np.int_)):
            self.trackcpp_acc.lattice.erase(
                self.trackcpp_acc.lattice.begin() + int(index))
        elif isinstance(index, (list, tuple, _np.ndarray)):
            for i in index:
                self.trackcpp_acc.lattice.erase(
                    self.trackcpp_acc.lattice.begin() + int(i))
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.trackcpp_acc.lattice))
            iterator = range(start, stop, step)
            for i in iterator:
                self.trackcpp_acc.lattice.erase(
                    self.trackcpp_acc.lattice.begin() + i)

    def __getitem__(self, index):
        if isinstance(index, (int, _np.int_)):
            ele = _elements.Element()
            ele.trackcpp_e = self.trackcpp_acc.lattice[int(index)]
            return ele
        elif isinstance(index, (list, tuple, _np.ndarray)):
            try:
                index = _np.array(index, dtype=int)
            except TypeError:
                raise TypeError('invalid index')
            lattice = _trackcpp.CppElementVector()
            for i in index:
                lattice.append(self.trackcpp_acc.lattice[int(i)])
        elif isinstance(index, slice):
            lattice = self.trackcpp_acc.lattice[index]
        else:
            raise TypeError('invalid index')
        acc = Accelerator(
            lattice=lattice,
            energy=self.trackcpp_acc.energy,
            harmonic_number=self.trackcpp_acc.harmonic_number,
            cavity_on=self.trackcpp_acc.cavity_on,
            radiation_on=self.trackcpp_acc.radiation_on,
            vchamber_on=self.trackcpp_acc.vchamber_on)
        return acc

    def __setitem__(self, index, value):
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
        return self.trackcpp_acc.lattice.size()

    def __str__(self):
        r = ''
        r += 'energy         : ' + str(self.trackcpp_acc.energy) + ' eV'
        r += '\nharmonic_number: ' + str(self.trackcpp_acc.harmonic_number)
        r += '\ncavity_on      : ' + str(self.trackcpp_acc.cavity_on)
        r += '\nradiation_on   : ' + str(self.trackcpp_acc.radiation_on)
        r += '\nvchamber_on    : ' + str(self.trackcpp_acc.vchamber_on)
        r += '\nlattice size   : ' + str(len(self.trackcpp_acc.lattice))
        r += '\nlattice length : ' + str(self.length) + ' m'
        return r

    def __add__(self, other):
        if isinstance(other, _elements.Element):
            a = self[:]
            a.append(other)
            return a
        elif isinstance(other, Accelerator):
            a = self[:]
            for e in other:
                a.append(e)
            return a
        else:
            msg = "unsupported operand type(s) for +: '" + \
                    self.__class__.__name__ + "' and '" + \
                    other.__class__.__name__ + "'"
            raise TypeError(msg)

    def __rmul__(self, other):
        if isinstance(other, (int, _np.int_)):
            if other < 0:
                raise ValueError('cannot multiply by negative integer')
            elif other == 0:
                return Accelerator(
                    energy=self.energy,
                    harmonic_number=self.harmonic_number,
                    cavity_on=self.cavity_on,
                    radiation_on=self.radiation_on,
                    vchamber_on=self.vchamber_on)
            else:
                a = self[:]
                other -= 1
                while other > 0:
                    a += self[:]
                    other -= 1
                return a
        else:
            msg = "unsupported operand type(s) for +: '" + \
                    other.__class__.__name__ + "' and '" + \
                    self.__class__.__name__ + "'"
            raise TypeError(msg)

    def __eq__(self, other):
        if not isinstance(other, Accelerator):
            return NotImplemented
        return self.trackcpp_acc.isequal(other.trackcpp_acc)

    # to make the class objects pickalable:
    def __getstate__(self):
        stri = _trackcpp.String()
        _trackcpp.write_flat_file_wrapper(stri, self.trackcpp_acc, False)
        return stri.data

    def __setstate__(self, stridata):
        stri = _trackcpp.String(stridata)
        acc = Accelerator()
        _trackcpp.read_flat_file_wrapper(stri, acc.trackcpp_acc, False)
        self.trackcpp_acc = acc.trackcpp_acc

    def pop(self, index):
        elem = self[index]
        del self[index]
        return elem

    def append(self, value):
        if not isinstance(value, _elements.Element):
            raise TypeError('value must be Element')
        self.trackcpp_acc.lattice.append(value.trackcpp_e)

    def extend(self, value):
        if not isinstance(value, Accelerator):
            raise TypeError('value must be Accelerator')
        if value is self:
            value = Accelerator(accelerator=value)
        for el in value:
            self.append(el)

    @property
    def length(self):
        """Lattice length in m"""
        return self.trackcpp_acc.get_length()

    @property
    def energy(self):
        """Beam energy in eV"""
        return self.trackcpp_acc.energy

    @energy.setter
    def energy(self, value):
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(energy=value/1e9)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def gamma_factor(self):
        return self._gamma

    @gamma_factor.setter
    def gamma_factor(self, value):
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(gamma=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def beta_factor(self):
        return self._beta

    @beta_factor.setter
    def beta_factor(self, value):
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(beta=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def velocity(self):
        """Beam velocity in m/s"""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(velocity=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def brho(self):
        return self._brho

    @brho.setter
    def brho(self, value):
        self._brho, self._velocity, self._beta, self._gamma, energy = \
            _mp.beam_optics.beam_rigidity(brho=value)
        self.trackcpp_acc.energy = energy * 1e9

    @property
    def harmonic_number(self):
        return self.trackcpp_acc.harmonic_number

    @harmonic_number.setter
    def harmonic_number(self, value):
        if not isinstance(value, int) or value < 1:
            raise AcceleratorException(
                'harmonic number has to be a positive integer')
        self.trackcpp_acc.harmonic_number = value

    @property
    def cavity_on(self):
        return self.trackcpp_acc.cavity_on

    @cavity_on.setter
    def cavity_on(self, value):
        if self.trackcpp_acc.harmonic_number < 1:
            raise AcceleratorException('invalid harmonic number')
        self.trackcpp_acc.cavity_on = value

    @property
    def radiation_on(self):
        return self.trackcpp_acc.radiation_on

    @radiation_on.setter
    def radiation_on(self, value):
        self.trackcpp_acc.radiation_on = value

    @property
    def vchamber_on(self):
        return self.trackcpp_acc.vchamber_on

    @vchamber_on.setter
    def vchamber_on(self, value):
        self.trackcpp_acc.vchamber_on = value
