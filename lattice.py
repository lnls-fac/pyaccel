
import numpy as _numpy
import mathphys as _mp
import trackcpp as _trackcpp
import pyaccel as _pyaccel
from pyaccel.utils import interactive as _interactive


class LatticeException(Exception):
    pass


@_interactive
def flatlat(elist):
    """Take a list-of-list-of-... elements and flattens it: a simple list of lattice elements"""
    flat_elist = []
    for element in elist:
        try:
            famname = element.fam_name
            flat_elist.append(element)
        except:
            flat_elist.extend(flatlat(element))
    return flat_elist


@_interactive
def buildlat(elist):
    """Build lattice from a list of elements and lines"""
    lattice = _trackcpp.CppElementVector()
    elist = flatlat(elist)
    for e in elist:
        lattice.append(e._e)
    return lattice


@_interactive
def shiftlat(lattice, start):
    """Shift periodically the lattice so that it starts at element whose index is 'start'"""
    new_lattice = lattice[start:]
    for i in range(start):
        new_lattice.append(lattice[i])
    return new_lattice


@_interactive
def lengthlat(lattice):
    length = [e.length for e in lattice]
    return sum(length)


@_interactive
def findspos(lattice, indices=None):
    """Return longitudinal position of the entrance for all lattice elements"""
    length = [0] + [e.length for e in lattice]
    pos = _numpy.cumsum(length)

    if indices is None or indices == 'open':
        return pos[:-1]
    elif indices == 'closed':
        return pos
    elif isinstance(indices, int):
        return pos[indices]
    else:
        return pos[list(indices)]


@_interactive
def findcells(lattice, attribute_name, value=None):
    """Returns a list with indices of elements that match criteria 'attribute_name=value'"""
    indices = []
    for i in range(len(lattice)):
        if hasattr(lattice[i], attribute_name):
            if value == None:
                if getattr(lattice[i], attribute_name) != None:
                    indices.append(i)
            else:
                if _is_equal(getattr(lattice[i], attribute_name), value):
                    indices.append(i)
    return indices


@_interactive
def getcellstruct(lattice, attribute_name, indices = None, m=None, n=None):
    """Return a list with requested lattice data"""
    if indices is None:
        indices = range(len(lattice))
    else:
        try:
            indices[0]
        except:
            indices = [indices]

    data = []
    for idx in indices:
        tdata = getattr(lattice[idx], attribute_name)
        if n is None:
            if m is None:
                data.append(tdata)
            else:
                data.append(tdata[m])
        else:
            if m is None:
                data.append(tdata)
            else:
                data.append(tdata[m][n])
    return data


@_interactive
def setcellstruct(lattice, attribute_name, indices, values):
    """ sets elements data and returns a new updated lattice """
    try:
        indices[0]
    except:
        indices = [indices]

    for idx in range(len(indices)):
        if isinstance(values, (tuple, list)):
            try:
                isinstance(values[0],(tuple,list,_numpy.ndarray))
                if len(values) == 1:
                    values=[values[0]]*len(indices)
                setattr(lattice[indices[idx]], attribute_name, values[idx])
            except:
                setattr(lattice[indices[idx]], attribute_name, values[idx])
        else:
            setattr(lattice[indices[idx]], attribute_name, values)
    return lattice


@_interactive
def finddict(lattice, attribute_name):
    """Return a dict which correlates values of 'attribute_name' and a list of indices corresponding to matching elements"""
    latt_dict = {}
    for i in range(len(lattice)):
        if hasattr(lattice[i], attribute_name):
            att_value = getattr(lattice[i], attribute_name)
            if att_value in latt_dict:
                latt_dict[att_value].append(i)
            else:
                latt_dict[att_value] = [i]
    return latt_dict


@_interactive
def knobvalue_set(lattice, fam_name, attribute_name, value):

    if isinstance(fam_name,str):
        idx = findcells(lattice, 'fam_name', fam_name)
    else:
        idx = []
        for famname in fam_name:
            idx.append(findcells(lattice, 'fam_name', fam_name))
    for i in idx:
        setattr(lattice[i], attribute_name, value)

@_interactive
def knobvalue_add(lattice, fam_name, attribute_name, value):

    if isinstance(fam_name,str):
        idx = findcells(lattice, 'fam_name', fam_name)
    else:
        idx = []
        for famname in fam_name:
            idx.append(findcells(lattice, 'fam_name', fam_name))
    for i in idx:
        original_value = getattr(lattice[i], attribute_name)
        new_value = original_values + value
        setattr(lattice[i], attribute_name, new_value)

def _is_equal(a,b):
    # checks for strings
    if isinstance(a,str):
        if isinstance(b,str):
            return a == b
        else:
            return False
    else:
        if isinstance(b,str):
            return False
    try:
        a[0]
        # 'a' is an iterable
        try:
            b[0]
            # 'b' is an iterable
            if len(a) != len(b):
                # 'a' and 'b' are iterbales but with different lengths
                return False
            else:
                # 'a' and 'b' are iterables with the same length
                for i in range(len(a)):
                    if not _is_equal(a[i],b[i]):
                        return False
                # corresponding elements in a and b iterables are the same.
                return True
        except:
            # 'a' is iterable but 'b' is not
            return False
    except:
        # 'a' is not iterable
        try:
            b[0]
            # 'a' is not iterable but 'b' is.
            return False
        except:
            # neither 'a' nor 'b' are iterables
            return a == b

@_interactive
def read_flat_file(filename):
    e = _mp.constants.electron_rest_energy*_mp.units.joule_2_eV
    a = _pyaccel.accelerator.Accelerator(energy=e) # energy cannot be zero
    r = _trackcpp.read_flat_file(filename, a._accelerator)
    if r > 0:
        raise LatticeException(_trackcpp.string_error_messages[r])

    return a

@_interactive
def write_flat_file(accelerator, filename):
    r = _trackcpp.write_flat_file(filename, accelerator._accelerator)
    if r > 0:
        raise LatticeException(_trackcpp.string_error_messages[r])
