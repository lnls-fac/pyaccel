
import os as _os
from . import elements
from . import accelerator
from . import lattice
from . import tracking
from . import optics


with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
