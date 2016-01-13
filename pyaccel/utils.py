
import numpy as _numpy


interactive_list = []


def interactive(obj):
    interactive_list.append(
        {
            'module': obj.__module__,
            'name': obj.__name__
        }
    )

    return obj


class Polynom(_numpy.ndarray):

    def __new__(cls, polynom):
        shape = (len(polynom),)
        array = _numpy.ndarray.__new__(cls, shape=shape)
        array[:] = polynom[:]
        array._polynom = polynom
        return array

    def __setitem__(self, index, value):
        if hasattr(self, '_polynom'):
            self._polynom[index] = value
        super().__setitem__(index, value)

    def __eq__(self,other):
        if not isinstance(other,Polynom): return NotImplemented
        if len(self) != len(other): return False
        if (self != other).any(): return False
        return True
