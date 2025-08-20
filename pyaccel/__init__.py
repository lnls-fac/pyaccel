"""Pyaccel package."""

# modules
from . import elements
from . import accelerator
from . import lattice
from . import tracking
from . import graphics
from . import intrabeam_scattering
from . import lifetime
from . import naff
from . import utils

# subpackages
from . import optics

import os as _os
with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()

__all__ = ['elements', 'accelerator', 'lattice', 'tracking', 'graphics',
           'intrabeam_scattering', 'lifetime', 'naff', 'utils', '__version__']

__all__.extend(['optics'])
