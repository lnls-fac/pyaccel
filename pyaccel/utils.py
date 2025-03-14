"""."""

import warnings

import trackcpp as _trackcpp

INTERACTIVE_LIST = []

DISTRIBUTIONS_NAMES = tuple(_trackcpp.distributions_dict)


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
    """Set distribution of random numbers used to simulate quantum excitation.

    Args:
        distribution (str|int): distribution to be used.
            - 0 or 'normal': Normal distribution,
            - 1 or 'uniform': Uniform distribution.

    Raises:
        ValueError
    """
    dist = distribution
    if isinstance(dist, int) and (dist < len(DISTRIBUTIONS_NAMES)):
        _trackcpp.set_random_distribution(dist)
    elif isinstance(dist, str) and (dist in DISTRIBUTIONS_NAMES):
        _trackcpp.set_random_distribution(DISTRIBUTIONS_NAMES.index(dist))
    else:
        errstr = 'The distribution must be one of the options: ' + \
            f"{DISTRIBUTIONS_NAMES}"
        raise ValueError(errstr)
