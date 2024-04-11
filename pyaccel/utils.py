"""."""

import warnings

import numpy as _numpy

from mathphys.functions import get_namedtuple as _get_namedtuple


# # Distributions = _trackcpp.distributions_dict
# DISTRIBUTIONS_NAMES = _trackcpp.distributions_dict
# _states_str = tuple(state.upper() for state in _trackcpp.distributions_dict)
# _indices = (idx for idx in range(len(_states_str)))
# DISTRIBUTIONS_NAMES = _get_namedtuple('DISTRIBUTIONS_NAMES',
#     _states_str, _indices)
# del(_states_str)
# del(_indices)

# RADIATION_STATES_NAMES = _trackcpp.rad_dict
# _states_str = tuple(state.upper() for state in _trackcpp.rad_dict)
# _indices = (idx for idx in range(len(_states_str)))
# RADIATION_STATES = _get_namedtuple('RADIATION_STATES', _states_str, _indices)
# del(_states_str)
# del(_indices)

INTERACTIVE_LIST = []

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


# @interactive
# def set_random_seed(rnd_seed : int):
#     """Set random number seed used in trackcpp."""
#     _trackcpp.set_random_seed(rnd_seed)


# @interactive
# def get_random_number():
#     """Return a random number from trackcpp normal distribution."""
#     return _trackcpp.gen_random_number()


# @interactive
# def set_distribution(distribution):
#     """Sets the distribution of the random numbers used to simulate quantum
#     excitation effects.

#     Args:
#         distribution (str or int):
#         - 0 or 'normal': Normal distribution,
#         - 1 or 'uniform': Uniform distribution.

#     Raises:
#         ValueError
#     """
#     dist = distribution
#     if isinstance(dist, int) and (dist < len(DISTRIBUTIONS_NAMES)):
#         _trackcpp.set_random_distribution(dist)
#     elif isinstance(dist, str) and (dist in DISTRIBUTIONS_NAMES):
#         _trackcpp.set_random_distribution(DISTRIBUTIONS_NAMES.index(dist))
#     else:
#         errstr = (
#             'The distribution must be one of the options: '
#             f'{DISTRIBUTIONS_NAMES}',
#             )
#         raise ValueError(errstr)
