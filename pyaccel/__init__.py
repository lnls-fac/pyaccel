"""Pyaccel package."""

from .backends import Backend as _Backend

global _backend
_backend = _Backend()

def set_backend(backend:str):
    """Change the backend.

    Args:
        backend (str): backend name. Should be: "cpp" or "julia".
    """
    if backend not in ["cpp", "julia"]:
        raise ValueError("Invalid backend. Choose between \"cpp\" and \"julia\".")
    import importlib
    module = importlib.import_module("pyaccel.backends._"+backend)
    global _backend
    vars(_backend).update(vars(module))

set_backend("cpp")

def get_backend():
    return _backend

from . import elements
from . import accelerator
from . import lattice
# from . import tracking
# from . import graphics
# from . import intrabeam_scattering
# from . import lifetime
# from . import naff
# from . import utils

import os as _os
with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
