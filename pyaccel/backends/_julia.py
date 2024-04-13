"""."""

import numpy as _numpy
from juliacall import Main as _jl
_trjulia = _jl.seval("using Track; Track")
_jl.seval("""
using PythonCall
function PythonCall.JlWrap.pyjl_attr_py2jl(k::String)
    if k == "polynom_b"
        return k
    else
        return replace(k, r"_[b]+$" => (x -> "!"^(length(x) - 1)))
    end
end
""")

_NUM_COORDS   = 6
_DIMS         = (_NUM_COORDS, _NUM_COORDS)

language  = "julia"
Element       = _trjulia.Elements.Element
PASS_METHODS  = _trjulia.Tracking.pm_dict
I_SHIFT       = 1
float_MAX     = _jl.Inf
_COORD_VECTOR = None
_COORD_MATRIX = None

Accelerator   = _trjulia.AcceleratorModule.Accelerator
ElementVector = _jl.Vector[_trjulia.Elements.Element]

marker        = _trjulia.Elements.marker
bpm           = _trjulia.Elements.bpm
drift         = _trjulia.Elements.drift
matrix        = _trjulia.Elements.matrix
hcorrector    = _trjulia.Elements.hcorrector
vcorrector    = _trjulia.Elements.vcorrector
corrector     = _trjulia.Elements.corrector
rbend         = _trjulia.Elements.rbend
quadrupole    = _trjulia.Elements.quadrupole
sextupole     = _trjulia.Elements.sextupole
rfcavity      = _trjulia.Elements.rfcavity
kickmap       = _trjulia.Elements.kickmap


def PassMethod(index):
    return _trjulia.Auxiliary.PassMethod(index)

def Int(value):
    return int(_jl.Int(value))

def FloatVector(arr=None):
    return _jl.Vector[_jl.Float64](arr)

def VChamberShape(value):
    return _trjulia.Auxiliary.VChamberShape(int(value))

def get_array(pointer):
    return _numpy.array(pointer)

def get_matrix(pointer):
    return _numpy.array(pointer).reshape(_DIMS)

def set_array_from_vector(array, size, values):
    if not size == len(values):
        raise ValueError("array and vector must have same size")
    for i in range(size):
        _jl.setindex_b(array, values[i], i+I_SHIFT)

def set_array_from_matrix(array, shape, values):
    """."""
    if not shape == values.shape:
        raise ValueError("array and matrix must have same shape")
    rows, cols = shape
    for i in range(rows):
        for j in range(cols):
            _jl.setindex_b(
                array,
                values[i, j],
                (i*cols)+j+I_SHIFT
            )

def force_set(obj, field, value):
    _jl.setfield_b(obj, _jl.Symbol(field), value)

def get_size(obj):
    #size in cpp is equal length in julia
    return _jl.length(obj)

def get_acc_length(acc):
    return acc.length

def bkd_isinstance(obj, class_or_tuple):
    if isinstance(class_or_tuple, tuple):
        return any([_jl.isa(obj, class_) for class_ in class_or_tuple])
    else:
        return _jl.isa(obj, class_or_tuple)

def isequal(this, other):
    return _jl.isequal(this, other)

def get_kicktable(index):
    return _trjulia.Elements.kicktable_global_list[index]

def insertElement(lattice, element, index):
    _jl.insert_b(lattice, index+I_SHIFT, element)

def set_polynom(polynom, val):
    polynom = _jl.Vector[_jl.Float64](val)

def matrix66_set_by_index(matrix, row, column, value):
    matrix[row, column] = value

def matrix66_get_by_index(matrix,  row, column):
    return matrix[row, column]
