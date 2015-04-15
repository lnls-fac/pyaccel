
"""Interactive pyaccel module

Use this module to define variables and functions to be globally available when
using

    'from pyaccel.interactive import *'

Names starting with an underscore ('_') will not be exported. In addition to
the objects defined here, all objects in pyaccel with the '@interactive'
decorator will also be available. In order to use this decorator in a pyaccel
module, import it from pyaccel.utils with

    'from pyaccel.utils import interactive'
"""

import numpy as np
import matplotlib.pyplot as plt
import pyaccel as _pyaccel
import sirius.SI_V07 as _sirius_si


plt.ion()

create_accelerator = _sirius_si.create_accelerator

rx = 0
px = 1
ry = 2
py = 3
dl = 4
de = 5

__all__ = [name for name in dir() if not name.startswith('_')]

for f in _pyaccel.utils.interactive_list:
    name = f['name']
    module = getattr(_pyaccel, f['module'].split('.')[1])
    globals()[name] = getattr(module, name)
    __all__.append(name)

print('Names defined in pyaccel.interactive: ' + ', '.join(__all__) + '.\n')
print('Function create_accelerator from ' + _sirius_si.__name__ + '.')
