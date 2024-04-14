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

LOST_PLANES = (None, 'x', 'y', 'z', 'xy')  # See trackcpp.Plane

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

def Numpy2Pos(arr):
    if arr.shape[1] == 1:
        arr = arr.ravel()
        return _trjulia.Pos(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5])
    else:
        vec = _jl.Vector[_trjulia.Pos[_jl.Float64]]()
        for i in range(arr.shape[1]):
            vec.append(
                _trjulia.Pos(arr[0,i], arr[1,i], arr[2,i], arr[3,i], arr[4,i], arr[5,i])
            )
        return vec

def Pos2Numpy(pos):
    if _jl.isa(pos, _trjulia.Pos[_jl.Float64]):
        arr = _numpy.zeros((6,1))
        arr[0, 0] = pos.rx
        arr[1, 0] = pos.px
        arr[2, 0] = pos.ry
        arr[3, 0] = pos.py
        arr[4, 0] = pos.de
        arr[5, 0] = pos.dl
    elif _jl.isa(pos, _jl.Vector[_trjulia.Pos[_jl.Float64]]):
        lenvecpos = _jl.length(pos)
        arr = _numpy.zeros((6, lenvecpos))
        for i in range(lenvecpos):
            arr[0, i] = pos[i].rx
            arr[1, i] = pos[i].px
            arr[2, i] = pos[i].ry
            arr[3, i] = pos[i].py
            arr[4, i] = pos[i].de
            arr[5, i] = pos[i].dl
    elif _jl.isa(pos, _jl.Vector[_jl.Vector[_trjulia.Pos[_jl.Float64]]]):
        lenvecvecpos = _jl.length(pos)
        lenvecpos = _jl.length(pos[0])
        arr = _numpy.zeros((6, lenvecvecpos, lenvecpos))
        for i in range(lenvecvecpos):
            for j in range(lenvecpos):
                arr[0, i, j] = pos[i][j].rx
                arr[1, i, j] = pos[i][j].px
                arr[2, i, j] = pos[i][j].ry
                arr[3, i, j] = pos[i][j].py
                arr[4, i, j] = pos[i][j].de
                arr[5, i, j] = pos[i][j].dl
    else:
        raise ValueError
    return arr

def element_pass(element, p_in, accelerator):
    pos_in = Numpy2Pos(p_in)
    ret = _trjulia.Tracking.element_pass(
        element.backend_e, pos_in, accelerator.backend_acc
    )

    if _jl.isa(ret, _jl.Vector):
        ret = any([i == _trjulia.Auxiliary.st_success for i in ret])
    else:
        ret = ret == _trjulia.Auxiliary.st_success

    return ret, Pos2Numpy(pos_in)

def line_pass(accelerator, p_in, indices, element_offset):
    pos_in = Numpy2Pos(p_in)
    _indices = _jl.Vector[_jl.Int]([i+1 for i in indices])
    tracked_pos, status, lostelement, lostplane = _trjulia.Tracking.line_pass(
        accelerator.backend_acc,
        pos_in,
        _indices,
        element_offset=element_offset+1
        )
    p_out = Pos2Numpy(tracked_pos)
    if p_out.shape[1] == 1:
        return p_out, status!=_trjulia.Auxiliary.st_success, \
        [lostelement], [LOST_PLANES[_jl.Int(lostplane)]]
    else:
        return p_out, not all([st==_trjulia.Auxiliary.st_success for st in status]), \
            [int(i) for i in lostelement], [LOST_PLANES[_jl.Int(lp)] for lp in lostplane]
