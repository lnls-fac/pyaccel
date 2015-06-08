
import numpy as _numpy
import mathphys as _mp
import trackcpp as _trackcpp
import pyaccel as _pyaccel
import math as _math
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
    """Shift periodically the lattice so that it starts at element whose index
    is 'start'.

    Keyword arguments:
    lattice -- a list of objects
    start -- index of first element in new list

    Returns a list (not an Accelerator).
    """
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
def getattributelat(lattice, attribute_name, indices = None, m=None, n=None):
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
def setattributelat(lattice, attribute_name, indices, values):
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

@_interactive
def refine_lattice(accelerator,
                   max_length=None,
                   indices=None,
                   fam_names=None,
                   pass_methods=None):

    if max_length is None:
        max_length = 0.05

    acc = accelerator[:]

    # builds list with indices of elements to be affected
    if indices is None:
        indices = []
        # adds specified fam_names
        if fam_names is not None:
            for fam_name in fam_names:
                indices.extend(findcells(acc, 'fam_name', fam_name))
        # adds specified pass_methods
        if pass_methods is not None:
            for pass_method in pass_methods:
                indices.extend(findcells(acc, 'pass_method', pass_method))
        if fam_names is None and pass_methods is None:
            indices = list(range(len(acc)))

    new_accelerator = _pyaccel.accelerator.Accelerator(
        energy = acc.energy,
        harmonic_number = acc.harmonic_number,
        cavity_on = acc.cavity_on,
        radiation_on = acc.radiation_on,
        vchamber_on = acc.vchamber_on)

    for i in range(len(acc)):
        if i in indices:
            if acc[i].length <= max_length:
                new_accelerator.append(acc[i])
            else:

                nr_segs = 1+int(acc[i].length/max_length)

                if (acc[i].angle_in != 0) or (acc[i].angle_out != 0):
                    # for dipoles (special case due to fringe fields)
                    #new_accelerator.append(acc[i])
                    #break

                    nr_segs = max(3,nr_segs)
                    length  = acc[i].length
                    angle   = acc[i].angle

                    e     = _pyaccel.elements.Element(element = acc[i], copy = True)
                    e_in  = _pyaccel.elements.Element(element = acc[i], copy = True)
                    e_out = _pyaccel.elements.Element(element = acc[i], copy = True)

                    e_in.angle_out, e.angle_out, e.angle_in, e_out.angle_in = 4*(0,)
                    e_in.length, e.length, e_out.length = 3*(length/nr_segs,)
                    e_in.angle, e.angle, e_out.angle = 3*(angle/nr_segs,)

                    new_accelerator.append(e_in)
                    for k in range(nr_segs-2):
                        new_accelerator.append(e)
                    new_accelerator.append(e_out)
                elif acc[i].kicktable is not None:
                    raise Exception('no refinement implemented for IDs yet')
                else:
                    e = _pyaccel.elements.Element(element = acc[i]._e)
                    e.length = e.length / nr_segs
                    e.angle  = e.angle / nr_segs
                    for k in range(nr_segs):
                        new_accelerator.append(e)

        else:
            new_accelerator.append(acc[i])

    return new_accelerator


@_interactive
def add_error_misalignment_x(lattice, indices, values):
    """Add horizontal misalignment errors to lattice"""

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(indices.shape[0]):
        error = values[i]
        for j in range(indices.shape[1]):
            index = indices[i,j]
            lattice[index].t_in[0]  -=  values[i]
            lattice[index].t_out[0] -= -values[i]

    return lattice

@_interactive
def add_error_misalignment_y(lattice, indices, values):
    """Add vertical misalignment errors to lattice"""

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(indices.shape[0]):
        error = values[i]
        for j in range(indices.shape[1]):
            index = indices[i,j]
            lattice[index].t_in[2]  -=  values[i]
            lattice[index].t_out[2] -= -values[i]

    return lattice

@_interactive
def add_error_rotation_roll(lattice, indices, values):
    """Add roll rotation errors to lattice"""

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(indices.shape[0]):
        c, s = _math.cos(values[i]), _math.sin(values[i])
        rot = _numpy.diag([c,c,c,c,1.0,1.0])
        rot[0,2], rot[1,3], rot[2,0], rot[3,1] = s, s, -s, -s

        for j in range(indices.shape[1]):
            index = indices[i,j]
            if lattice[index].angle != 0 and lattice[index].length != 0:
                rho    = lattice[index].length / lattice[index].angle
                orig_s = lattice[index].polynom_a[0] * rho
                orig_c = lattice[index].polynom_b[0] * rho + 1.0  # look at bndpolysymplectic4pass
                lattice[index].polynom_a[0] = (orig_s * c + orig_c * s) / rho     # sin(teta)/rho
                lattice[index].polynom_b[0] = ((orig_c*c - orig_s*s) - 1.0) / rho # (cos(teta)-1)/rho
            else:
                lattice[index].r_in  = _numpy.dot(rot, lattice[index].r_in)
                lattice[index].r_out = _numpy.dot(lattice[index].r_out, rot.transpose())
    return lattice


def _process_args_errors(indices, values):
    if isinstance(indices,int):
        indices = _numpy.array([[indices]])
    elif len(indices) == 1:
        indices = _numpy.array([indices]).transpose()
    if isinstance(values,(int,float)):
        values = values * _numpy.ones(indices.shape[0])
    return indices, values


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
