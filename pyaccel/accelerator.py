
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

    __isfrozen = False # this is used to prevent creation of new attributes

    def __init__(self, **kwargs):

        if 'accelerator' in kwargs:
            a = kwargs['accelerator']
            if isinstance(a,_trackcpp.Accelerator):
                self._accelerator = a # points to the same object in memory
            elif isinstance(a,Accelerator): # creates another object.
                self._accelerator = _trackcpp.Accelerator()
                self._accelerator.lattice = a._accelerator.lattice[:]
                self._accelerator.cavity_on = a.cavity_on
                self._accelerator.radiation_on = a.radiation_on
                self._accelerator.vchamber_on = a.vchamber_on
                self._accelerator.harmonic_number = a.harmonic_number
        else:
            self._accelerator = _trackcpp.Accelerator()
            self._accelerator.cavity_on = False
            self._accelerator.radiation_on = False
            self._accelerator.vchamber_on = False
            self._accelerator.harmonic_number = 0

        if 'lattice' in kwargs:
            lattice = kwargs['lattice']
            if isinstance(lattice, _trackcpp.CppElementVector):
                self._accelerator.lattice = lattice
            elif isinstance(lattice, list):
                for i in range(len(lattice)):
                    e = lattice[i]
                    self._accelerator.lattice.append(e._e)
            else:
                raise TypeError('values must be list of Element')

        if 'energy' in kwargs:
            self._accelerator.energy = kwargs['energy']
        if 'harmonic_number' in kwargs:
            self._accelerator.harmonic_number = kwargs['harmonic_number']
        if 'radiation_on' in kwargs:
            self._accelerator.radiation_on = kwargs['radiation_on']
        if 'cavity_on' in kwargs:
            self._accelerator.cavity_on = kwargs['cavity_on']
        if 'vchamber_on' in kwargs:
            self._accelerator.vchamber_on = kwargs['vchamber_on']

        if self._accelerator.energy == 0:
            self._brho, self._velocity, self._beta, self._gamma, self._accelerator.energy = _mp.beam_optics.beam_rigidity(gamma = 1.0)
        else:
            self._brho, self._velocity, self._beta, self._gamma, self._accelerator.energy = _mp.beam_optics.beam_rigidity(energy = self.energy)

        self.__isfrozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise AcceleratorException( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def __delitem__(self, index):
        if isinstance(index,(int,_np.int_)):
            self._accelerator.lattice.erase(self._accelerator.lattice.begin() + int(index));
        elif isinstance(index, (list,tuple)):
            for i in index:
                self._accelerator.lattice.erase(self._accelerator.lattice.begin() + i);
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self._accelerator.lattice))
            iterator = range(start, stop, step)
            for i in iterator:
                self._accelerator.lattice.erase(self._accelerator.lattice.begin() + i);

    def __getitem__(self, index):
        if isinstance(index,(int, _np.int_)):
            return _elements.Element(element=self._accelerator.lattice[int(index)])
        elif isinstance(index, (list,tuple,_np.ndarray)) and all(isinstance(x, (int, _np.int_)) for x in index):
            lattice = _trackcpp.CppElementVector()
            for i in index:
                lattice.append(self._accelerator.lattice[int(i)])
        elif isinstance(index, slice):
            lattice = self._accelerator.lattice[index]
        else:
            raise TypeError('invalid index')
        a = Accelerator(
                lattice=lattice,
                energy=self._accelerator.energy,
                harmonic_number=self._accelerator.harmonic_number,
                cavity_on=self._accelerator.cavity_on,
                radiation_on=self._accelerator.radiation_on,
                vchamber_on=self._accelerator.vchamber_on)
        return a

    def __setitem__(self, index, value):
        if isinstance(index, (int, _np.int_)):
            self._accelerator.lattice[int(index)] = value._e
        elif isinstance(index, (list, tuple)):
            if isinstance(value, (list, tuple, _np.ndarray,Accelerator)):
                for i in range(len(value)):
                    v = value[i]
                    if not isinstance(v, _elements.Element):
                        raise TypeError('invalid value')
                    self._accelerator.lattice[index[i]] = v._e
            else:
                if not isinstance(value, _elements.Element):
                    raise TypeError('invalid value')
                for i in range(len(value)):
                    self._accelerator.lattice[index[i]] = value._e
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self._accelerator.lattice))
            iterator = range(start, stop, step)
            if isinstance(value, (list, tuple, _np.ndarray, Accelerator)):
                for i, j in zip(iterator, range(len(iterator))):
                    self._accelerator.lattice[i] = value[j]._e
            else:
                for i in iterator:
                    self._accelerator.lattice[i] = value._e

    def __len__(self):
        return len(self._accelerator.lattice)

    def __str__(self):
        r = ''
        r +=   'energy         : ' + str(self._accelerator.energy) + ' eV'
        r += '\nharmonic_number: ' + str(self._accelerator.harmonic_number)
        r += '\ncavity_on      : ' + str(self._accelerator.cavity_on)
        r += '\nradiation_on   : ' + str(self._accelerator.radiation_on)
        r += '\nvchamber_on    : ' + str(self._accelerator.vchamber_on)
        r += '\nlattice size   : ' + str(len(self._accelerator.lattice))
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
        if isinstance(other, int):
            if other < 0:
                raise ValueError('cannot multiply by negative integer')
            elif other == 0:
                return Accelerator(
                        energy=self.energy,
                        harmonic_number=self.harmonic_number,
                        cavity_on=self.cavity_on,
                        radiation_on=self.radiation_on,
                        vchamber_on=self.vchamber_on
                )
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

    def __eq__(self,other):
        if not isinstance(other,Accelerator): return NotImplemented
        return self._accelerator.isequal(other._accelerator)
        # if len(self) != len(other): return False
        # for i in range(len(self)):
        #     if self[i] != other[i]: return False
        # return True


    def pop(self, index):
        elem = self[index]
        del self[index]
        return elem

    def append(self, value):
        if not isinstance(value, _elements.Element):
            raise TypeError('value must be Element')
        self._accelerator.lattice.append(value._e)

    def extend(self,value):
        if not isinstance(value,Accelerator):
            raise TypeError('value must be Accelerator')
        if value is self: value = Accelerator(accelerator=value)
        for el in value:
            self.append(el)

    @property
    def length(self):
        """Lattice length in m"""
        return _lattice.length(self._accelerator.lattice)

    @property
    def energy(self):
        """Beam energy in eV"""
        return self._accelerator.energy

    @energy.setter
    def energy(self, value):
        self._brho, self._velocity, self._beta, \
        self._gamma, self._accelerator.energy = \
            _mp.beam_optics.beam_rigidity(energy = value)

    @property
    def gamma_factor(self):
        return self._gamma

    @gamma_factor.setter
    def gamma_factor(self, value):
        self._brho, self._velocity, self._beta, \
        self._gamma, self._accelerator.energy = \
            _mp.beam_optics.beam_rigidity(gamma = value)

    @property
    def beta_factor(self):
        return self._beta

    @beta_factor.setter
    def beta_factor(self, value):
        self._brho, self._velocity, self._beta, \
        self._gamma, self._accelerator.energy = \
            _mp.beam_optics.beam_rigidity(beta = value)

    @property
    def velocity(self):
        """Beam velocity in m/s"""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._brho, self._velocity, self._beta, \
        self._gamma, self._accelerator.energy = \
            _mp.beam_optics.beam_rigidity(velocity = value)

    @property
    def brho(self):
        return self._brho

    @brho.setter
    def brho(self, value):
        self._brho, self._velocity, self._beta, \
        self._gamma, self._accelerator.energy = \
            _mp.beam_optics.beam_rigidity(brho = value)

    @property
    def harmonic_number(self):
        return self._accelerator.harmonic_number

    @harmonic_number.setter
    def harmonic_number(self, value):
        if not isinstance(value, int) or value < 1:
            raise AcceleratorException('harmonic number has to be a positive integer')
        self._accelerator.harmonic_number = value

    @property
    def cavity_on(self):
        return self._accelerator.cavity_on

    @cavity_on.setter
    def cavity_on(self, value):
        if self._accelerator.harmonic_number < 1:
            raise AcceleratorException('invalid harmonic number')
        self._accelerator.cavity_on = value

    @property
    def radiation_on(self):
        return self._accelerator.radiation_on

    @radiation_on.setter
    def radiation_on(self, value):
        self._accelerator.radiation_on = value

    @property
    def vchamber_on(self):
        return self._accelerator.vchamber_on

    @vchamber_on.setter
    def vchamber_on(self, value):
        self._accelerator.vchamber_on = value
