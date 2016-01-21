
import warnings
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


def deprecated(function):
    '''Decorator for deprecated functions.'''
    def new_function(*args, **kwargs):
        warnings.warn(
            "call to deprecated function {}".format(function.__name__),
            category=DeprecationWarning
        )
        return function(*args, **kwargs)

    new_function.__name__ = function.__name__
    new_function.__doc__ = function.__doc__
    new_function.__dict__.update(function.__dict__)

    return new_function


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
