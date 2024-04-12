"""."""

import numpy as _numpy
import ctypes as _ctypes
import trackcpp as _trcpp

_NUM_COORDS   = 6
_DIMS         = (_NUM_COORDS, _NUM_COORDS)

language  = "cpp"
Element       = _trcpp.Element
PASS_METHODS  = _trcpp.pm_dict
I_SHIFT       = 0
float_MAX     = _trcpp.get_double_max()
_COORD_VECTOR = _ctypes.c_double*_NUM_COORDS
_COORD_MATRIX = _ctypes.c_double*_DIMS[0]*_DIMS[1]

Accelerator   = _trcpp.Accelerator
ElementVector = _trcpp.CppElementVector

marker        = _trcpp.marker_wrapper
bpm           = _trcpp.bpm_wrapper
drift         = _trcpp.drift_wrapper
matrix        = _trcpp.matrix_wrapper
hcorrector    = _trcpp.hcorrector_wrapper
vcorrector    = _trcpp.vcorrector_wrapper
corrector     = _trcpp.corrector_wrapper
rbend         = _trcpp.rbend_wrapper
quadrupole    = _trcpp.quadrupole_wrapper
sextupole     = _trcpp.sextupole_wrapper
rfcavity      = _trcpp.rfcavity_wrapper
kickmap       = _trcpp.kickmap_wrapper


def PassMethod(index):
    return index

def Int(value):
    return int(value)

def FloatVector(arr=None):
    return arr

def VChamberShape(value):
    return int(value)

def get_array(pointer):
    address = int(pointer)
    c_array = _COORD_VECTOR.from_address(address)
    return _numpy.ctypeslib.as_array(c_array)

def get_matrix(pointer):
    address = int(pointer)
    c_array = _COORD_MATRIX.from_address(address)
    return _numpy.ctypeslib.as_array(c_array)

def set_array_from_vector(array, size, values):
    if not size == len(values):
        raise ValueError("array and vector must have same size")
    for i in range(size):
        _trcpp.c_array_set(array, i, values[i])

def set_array_from_matrix(array, shape, values):
    """."""
    if not shape == values.shape:
        raise ValueError("array and matrix must have same shape")
    rows, cols = shape
    for i in range(rows):
        for j in range(cols):
            _trcpp.c_array_set(
                array,
                (i*cols)+j+I_SHIFT,
                values[i, j]
            )

def force_set(obj, field, value):
    setattr(obj, field, value)

def get_size(obj):
    return obj.size()

def get_acc_length(acc):
    return acc.get_length()

def bkd_isinstance(obj, class_or_tuple):
    return isinstance(obj, class_or_tuple)

def isequal(this, other):
    return this.isequal(other)

def get_kicktable(index):
    _trcpp.cvar.kicktable_list[index]

def insertElement(lattice, element, index):
    idx = lattice.begin()
    if index != 0:
        idx += int(index)
    lattice.insert(idx, element)

def set_polynom(polynom, val):
    polynom[:] = val[:]
