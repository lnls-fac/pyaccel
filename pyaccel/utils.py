"""."""

import warnings

import numpy as _numpy

import trackcpp as _trackcpp


INTERACTIVE_LIST = []
Distributions = _trackcpp.distributions_dict


def interactive(obj):
    """."""
    INTERACTIVE_LIST.append(
        {
            'module': obj.__module__,
            'name': obj.__name__
        }
    )

    return obj


def deprecated(function):
    """Define decorator for deprecated functions."""
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
    """."""

    def __new__(cls, polynom):
        """."""
        shape = (len(polynom),)
        array = _numpy.ndarray.__new__(cls, shape=shape)
        array[:] = polynom[:]
        array._polynom = polynom
        return array

    def __setitem__(self, index, value):
        """."""
        if hasattr(self, '_polynom'):
            self._polynom[index] = value
        super().__setitem__(index, value)

    def __eq__(self, other):
        """."""
        if not isinstance(other, Polynom):
            return NotImplemented
        if len(self) != len(other):
            return False
        if (self != other).any():
            return False
        return True


@interactive
def set_random_seed(rnd_seed : int):
    """Set random number seed used in trackcpp."""
    _trackcpp.set_random_seed(rnd_seed)


@interactive
def get_random_number():
    """Return a random number from trackcpp normal distribution."""
    return _trackcpp.gen_random_number()


@interactive
def set_distribution(distribution):
    """Sets the distribution of the random numbers used to simulate quantum
    excitation effects.

    Args:
        distribution (str or int):
        - 0 or 'normal': Normal distribution,
        - 1 or 'uniform': Uniform distribution.

    Raises:
        ValueError
    """
    if isinstance(distribution, int) and (distribution < len(Distributions)):
        _trackcpp.set_random_distribution(distribution)
    elif isinstance(distribution, str) and (distribution in Distributions):
        _trackcpp.set_random_distribution(Distributions.index(distribution))
    else:
        raise ValueError(
            f'The distribution must be one of the options: {Distributions}')

# class Matrix(_numpy.ndarray):

#     def __new__(cls, polynom):
#         shape = (len(polynom), len(polynom[0]))
#         array = _numpy.ndarray.__new__(cls, shape=shape)
#         for i, pol in enumerate(polynom):
#             array[i] = pol[:]
#         array._polynom = polynom
#         return array

#     def __setitem__(self, index, value):
#         if hasattr(self, '_polynom'):
#             self._polynom[index] = value
#         super().__setitem__(index, value)

#     def __eq__(self, other):
#         if not isinstance(other, Polynom):
#             return NotImplemented
#         if len(self) != len(other):
#             return False
#         if (self != other).any():
#             return False
#         return True
