
import math as _math
import numpy as _numpy
import mathphys as _mp
import trackcpp as _trackcpp
import pyaccel as _pyaccel
from pyaccel.utils import interactive as _interactive


class LatticeError(Exception):
    pass


@_interactive
def flatten(elist):
    """Take a list-of-list-of-... elements and flattens it: a simple list of lattice elements"""
    flat_elist = []
    for element in elist:
        try:
            famname = element.fam_name
            flat_elist.append(element)
        except:
            flat_elist.extend(flatten(element))
    return flat_elist


@_interactive
def build(elist):
    """Build lattice from a list of elements and lines"""
    lattice = _trackcpp.CppElementVector()
    elist = flatten(elist)
    for e in elist:
        lattice.append(e._e)
    return lattice


@_interactive
def shift(lattice, start):
    """Shift periodically the lattice so that it starts at element whose index
    is 'start'.

    Keyword arguments:
    lattice -- a list of objects
    start -- index of first element in new list

    Returns an Accelerator).
    """
    new_lattice = lattice[start:]
    for i in range(start):
        new_lattice.append(lattice[i])
    return new_lattice


@_interactive
def length(lattice):
    length = [e.length for e in lattice]
    return sum(length)


@_interactive
def find_spos(lattice, indices='open'):
    """Return longitudinal position of the entrance of lattice elements.

    INPUTS:
        lattice : accelerator model.
        indices : may be a string 'closed' or 'open' to return or not the position
                  at the end of the last element, or a list or tuple to select
                  some indices or even an integer. Default is 'open'

    """
    length = [0] + [e.length for e in lattice]
    pos = _numpy.cumsum(length)

    if indices.lower() == 'open':
        return pos[:-1]
    elif indices.lower() == 'closed':
        return pos
    elif isinstance(indices, int):
        return pos[indices]
    elif isinstance(indices,(list,tuple)):
        return pos[list(indices)]
    else:
        raise TypeError('indices type not supported')

@_interactive
def find_indices(lattice, attribute_name, value, comparison=None):
    """Returns a list with indices (i) of elements that match criteria
    'lattice[i].attribute_name == value'

    INPUTS:
        lattice : accelerator model.
        attribute_name : string identifying the attribute to match
        value   : can be any data type or collection data type.
        comparison: function which takes two arguments, the value of the
            attribute and value, performs a comparison between then and returns
            a boolean. The default is equality comparison.

    OUTPUTS: list of indices where the comparison returns True.

    EXAMPLES:
      >> mia_idx = find_indices(lattice,'fam_name',value='mia')
      >> idx = find_indices(lattice,'polynom_b',value=[0.0,1.5,0.0])
      >> fun = lambda x,y: x[2] != y
      >> sext_idx = find_indices(lattice,'polynom_b',value=0.0,comparison=fun)
      >> fun2=lambda x,y: x.startswith(y)
      >> mi_idx = find_indices(lattice,'fam_name',value='mi',comparison=fun2)
    """

    if comparison is None: comparison = _is_equal
    indices = []
    for i in range(len(lattice)):
        attrib = getattr(lattice[i], attribute_name)
        try:
            boo = comparison(attrib, value)
            if not isinstance(boo,(_numpy.bool_,bool)): raise TypeError
            if boo: indices.append(i)
        except TypeError:
            raise TypeError('Comparison must take two arguments and return boolean.')
    return indices


@_interactive
def get_attribute(lattice, attribute_name, indices=None, m=None, n=None):
    """Return a list with requested lattice data"""
    # Check whether we have an Accelerator object
    # if (hasattr(lattice, '_accelerator') and
    #         hasattr(lattice._accelerator, 'lattice')):
    #     lattice = lattice._accelerator.lattice

    if indices is None:
        indices = range(len(lattice))
    else:
        try:
            indices[0]
        except:
            indices = [indices]

    data = []
    # for idx in indices:
    #     tdata = getattr(lattice[idx], attribute_name)
    #     if n is None:
    #         if m is None:
    #             data.append(tdata)
    #         else:
    #             data.append(tdata[m])
    #     else:
    #         if m is None:
    #             data.append(tdata)
    #         else:
    #             data.append(tdata[m][n])
    if (m is not None) and (n is not None):
        for idx in indices:
            tdata = getattr(lattice[idx], attribute_name)
            data.append(tdata[m][n])
    elif (m is not None) and (n is None):
        for idx in indices:
            tdata = getattr(lattice[idx], attribute_name)
            data.append(tdata[m])
    else:
        # Check whether we have an Accelerator object
        if (hasattr(lattice, '_accelerator') and
                hasattr(lattice._accelerator, 'lattice') and
                hasattr(lattice._accelerator.lattice[0], attribute_name)):
            lattice = lattice._accelerator.lattice
        for idx in indices:
            tdata = getattr(lattice[idx], attribute_name)
            data.append(tdata)

    return data


@_interactive
def set_attribute(lattice, attribute_name, indices, values):
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


@_interactive
def find_dict(lattice, attribute_name):
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
def set_knob(lattice, fam_name, attribute_name, value):

    if isinstance(fam_name,str):
        idx = find_indices(lattice, 'fam_name', fam_name)
    else:
        idx = []
        for famname in fam_name:
            idx.append(find_indices(lattice, 'fam_name', fam_name))
    for i in idx:
        setattr(lattice[i], attribute_name, value)


@_interactive
def add_knob(lattice, fam_name, attribute_name, value):

    if isinstance(fam_name,str):
        idx = find_indices(lattice, 'fam_name', fam_name)
    else:
        idx = []
        for famname in fam_name:
            idx.append(find_indices(lattice, 'fam_name', fam_name))
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
        raise LatticeError(_trackcpp.string_error_messages[r])

    return a


@_interactive
def write_flat_file(accelerator, filename):
    r = _trackcpp.write_flat_file(filename, accelerator._accelerator)
    if r > 0:
        raise LatticeError(_trackcpp.string_error_messages[r])


@_interactive
def refine_lattice(accelerator,
                   max_length=None,
                   indices=None,
                   fam_names=None,
                   pass_methods=None):

    if max_length is None:
        max_length = 0.05

    acc = accelerator

    # builds list with indices of elements to be affected
    if indices is None:
        indices = []
        # adds specified fam_names
        if fam_names is not None:
            for fam_name in fam_names:
                indices.extend(find_indices(acc, 'fam_name', fam_name))
        # adds specified pass_methods
        if pass_methods is not None:
            for pass_method in pass_methods:
                indices.extend(find_indices(acc, 'pass_method', pass_method))
        if fam_names is None and pass_methods is None:
            indices = list(range(len(acc)))

    new_acc = _pyaccel.accelerator.Accelerator(
        energy = acc.energy,
        harmonic_number = acc.harmonic_number,
        cavity_on = acc.cavity_on,
        radiation_on = acc.radiation_on,
        vchamber_on = acc.vchamber_on)

    for i in range(len(acc)):
        if i in indices:
            if acc[i].length <= max_length:
                new_acc.append(acc[i])
            else:

                nr_segs = 1+int(acc[i].length/max_length)

                if (acc[i].angle_in != 0) or (acc[i].angle_out != 0):
                    # for dipoles (special case due to fringe fields)
                    #new_acc.append(acc[i])
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

                    new_acc.append(e_in)
                    for k in range(nr_segs-2):
                        new_acc.append(e)
                    new_acc.append(e_out)
                elif acc[i].kicktable is not None:
                    raise Exception('no refinement implemented for IDs yet')
                else:
                    e = _pyaccel.elements.Element(element = acc[i], copy = True)
                    e.length = e.length / nr_segs
                    e.angle  = e.angle / nr_segs
                    for k in range(nr_segs):
                        new_acc.append(e)

        else:
            e = _pyaccel.elements.Element(element = acc[i], copy = True)
            new_acc.append(e)

    return new_acc


@_interactive
def get_error_misalignment_x(lattice, indices):
    """Get horizontal misalignment errors from lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.

    Outputs:
       list of floats, in case len(indices)>1, or float of errors. Unit: [meters]
    """

    ''' processes arguments '''
    indices, *_ = _process_args_errors(indices, 0.0)

    ''' loops over elements and gets error from T_IN '''
    values = []
    for i in range(len(indices)):
        segs = indices[i]
        #it is possible to also have yaw errors,so:
        misx = -(lattice[segs[ 0]].t_in[0] - lattice[segs[-1]].t_out[0])/2
        values.extend(len(segs)*[misx])

    if len(values) == 1:
        return values[0]
    else:
        return values


@_interactive
def set_error_misalignment_x(lattice, indices, values):
    """Set (discard previous) horizontal misalignments errors to lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit [meters]
    """

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(len(indices)):
        segs = indices[i]
        #it is possible to also have yaw errors, so:
        yaw = (lattice[segs[0]].t_in[0] + lattice[segs[-1]].t_out[0])/2
        for j in range(len(segs)):
            idx = segs[j]
            lattice[idx].t_in[0]  = yaw - values[i]
            lattice[idx].t_out[0] = yaw + values[i]



@_interactive
def add_error_misalignment_x(lattice, indices, values):
    """Add (sum to previous) horizontal misalignment errors to lattice.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit: [meters]
    """

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(len(indices)):
        segs = indices[i]
        for j in range(len(indices[i])):
            idx = segs[j]
            lattice[idx].t_in[0]  += -values[i]
            lattice[idx].t_out[0] += +values[i]


@_interactive
def get_error_misalignment_y(lattice, indices):
    """Get vertical misalignment errors from lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.

    Outputs:
       list, in case len(indices)>1, or float of errors. Unit: [meters]
    """

    ''' processes arguments '''
    indices, *_ = _process_args_errors(indices, 0.0)

    ''' loops over elements and gets error from T_IN '''
    values = []
    for i in range(len(indices)):
        segs = indices[i]
        #it is possible to also have pitch errors,so:
        misy = -(lattice[segs[0]].t_in[2]- lattice[segs[-1]].t_out[2])/2
        values.extend(len(segs)*[misy])
    if len(values) == 1:
        return values[0]
    else:
        return values


@_interactive
def set_error_misalignment_y(lattice, indices, values):
    """Set (discard previous) vertical misalignments errors to lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit [meters].
    """

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(len(indices)):
        segs = indices[i]
        #it is possible to also have yaw errors, so:
        pitch = (lattice[segs[0]].t_in[2] + lattice[segs[-1]].t_out[2])/2
        for j in range(len(segs)):
            idx = segs[j]
            lattice[idx].t_in[2]  = pitch - values[i]
            lattice[idx].t_out[2] = pitch + values[i]


@_interactive
def add_error_misalignment_y(lattice, indices, values):
    """Add (sum to previous) vertical misalignment errors to lattice.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit: [meters]
    """

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(len(indices)):
        segs = indices[i]
        for j in range(len(segs)):
            idx = segs[j]
            lattice[idx].t_in[2]  += -values[i]
            lattice[idx].t_out[2] +=  values[i]

    return lattice


@_interactive
def get_error_rotation_roll(lattice, indices):
    """Get roll rotation errors from lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.

    Outputs:
       list, in case len(indices)>1, or float of roll errors. Unit: [rad]
    """

    ''' processes arguments '''
    indices, *_ = _process_args_errors(indices, 0.0)

    ''' loops over elements and gets error from R_IN '''
    values = []
    for i in range(len(indices)):
        segs = indices[i]
        for j in range(len(segs)):
            idx = segs[j]
            angle = _math.asin(lattice[index].r_in[0,2])
            values.append(angle)
    if len(values) == 1:
        return values[0]
    else:
        return values


@_interactive
def set_error_rotation_roll(lattice, indices, values):
    """Set (discard previous) roll rotation errors to lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit [rad].
    """

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its R1 and R2 fields '''
    for i in range(len(indices)):
        segs = indices[i]
        c, s = _math.cos(values[i]), _math.sin(values[i])
        rot = _numpy.diag([c,c,c,c,1.0,1.0])
        rot[0,2], rot[1,3], rot[2,0], rot[3,1] = s, s, -s, -s

        for j in range(len(segs)):
            idx = segs[j]
            if lattice[idx].angle != 0 and lattice[idx].length != 0:
                rho    = lattice[idx].length / lattice[idx].angle
                orig_s = lattice[idx].polynom_a[0] * rho
                orig_c = lattice[idx].polynom_b[0] * rho + 1.0  # look at bndpolysymplectic4pass
                lattice[idx].polynom_a[0] = (orig_s * c + orig_c * s) / rho     # sin(teta)/rho
                lattice[idx].polynom_b[0] = ((orig_c*c - orig_s*s) - 1.0) / rho # (cos(teta)-1)/rho
            else:
                lattice[idx].r_in  = rot
                lattice[idx].r_out = rot.T


@_interactive
def add_error_rotation_roll(lattice, indices, values):
    """Add (sum to previous) roll rotation errors to lattice.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit: [rad]
    """

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its R1 and R2 fields '''
    for i in range(len(indices)):
        segs = indices[i]
        c, s = _math.cos(values[i]), _math.sin(values[i])
        rot = _numpy.diag([c,c,c,c,1.0,1.0])
        rot[0,2], rot[1,3], rot[2,0], rot[3,1] = s, s, -s, -s

        for j in range(len(segs)):
            idx = segs[j]
            if lattice[idx].angle != 0 and lattice[idx].length != 0:
                rho    = lattice[idx].length / lattice[idx].angle
                orig_s = lattice[idx].polynom_a[0] * rho
                orig_c = lattice[idx].polynom_b[0] * rho + 1.0  # look at bndpolysymplectic4pass
                lattice[idx].polynom_a[0] = (orig_s * c + orig_c * s) / rho     # sin(teta)/rho
                lattice[idx].polynom_b[0] = ((orig_c*c - orig_s*s) - 1.0) / rho # (cos(teta)-1)/rho
            else:
                lattice[idx].r_in  = _numpy.dot(rot, lattice[idx].r_in)
                lattice[idx].r_out = _numpy.dot(lattice[idx].r_out, rot.T)


@_interactive
def get_error_rotation_pitch(lattice, indices):
    """Get pitch rotation errors of lattice elements

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.

    Outputs:
       list, in case len(indices)>1, or float of pitch errors. Unit: [rad]
    """

    ''' processes arguments '''
    indices, *_ = _process_args_errors(indices, 0.0)

    ''' loops over elements and gets error from T_IN '''
    values = []
    for i in range(len(indices)):
        segs = indices[i]
        ang = lattice[segs[0]].t_in[3]
        values.extend(len(segs)*[-ang])

    if len(values) == 1: return values[0]
    else: return values


@_interactive
def set_error_rotation_pitch(lattice, indices, values):
    """Set (discard previous) pitch rotation errors to lattice.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit [rad]
    """

    #processes arguments
    indices, values = _process_args_errors(indices, values)

    #set new values to first T1 and last T2
    for i in range(len(indices)):
        segs = indices[i]
        angy = -values[i]
        L    = sum([lattice[ii].length for ii in segs])
        #It is possible that there is a misalignment error, so:
        misy = (lattice[segs[0]].t_in[2] - lattice[segs[-1]].t_out[2])/2

        # correction of the path length
        old_angx = lattice[segs[0]].t_in[1]
        path = -(L/2)*(angy*angy + old_angx*old_angx)

        #Apply the errors only to the entrance of the first and exit of the last segment:
        lattice[segs[ 0]].t_in[2]  = -(L/2)*angy+misy
        lattice[segs[-1]].t_out[2] = -(L/2)*angy-misy
        lattice[segs[ 0]].t_in[3]  =  angy
        lattice[segs[-1]].t_out[3] = -angy
        lattice[segs[-1]].t_out[5] =  path

@_interactive
def add_error_rotation_pitch(lattice, indices, values):
    """Add (sum to previous) pitch rotation errors to lattice.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit [rad]
    """

    #processes arguments
    indices, values = _process_args_errors(indices, values)

    #set new values to first T1 and last T2. Uses small angle approximation
    for i in range(len(indices)):
        segs = indices[i]
        angy  = -values[i]
        L    = sum([lattice[ii].length for ii in segs])

        # correction of the path length
        old_angy = lattice[segs[0]].t_in[3]
        path = -(L/2)*((angy+old_angy)*(angy+old_angy) - old_angy*old_angy)

        #Apply the errors only to the entrance of the first and exit of the last segment:
        lattice[segs[ 0]].t_in  += _numpy.array([0,0,-(L/2)*angy, angy,0,0])
        lattice[segs[-1]].t_out += _numpy.array([0,0,-(L/2)*angy,-angy,0,path])


@_interactive
def get_error_rotation_yaw(lattice, indices):
    """Get yaw rotation errors of lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.

    Outputs:
       list, in case len(indices)>1, or float of yaw errors. Unit: [rad]
    """

    ''' processes arguments '''
    indices, *_ = _process_args_errors(indices, 0.0)

    ''' loops over elements and gets error from T_IN '''
    values = []
    for i in range(len(indices)):
        segs = indices[i]
        ang = lattice[segs[0]].t_in[1]
        values.extend(len(segs)*[-ang])

    if len(values) == 1: return values[0]
    else: return values


@_interactive
def set_error_rotation_yaw(lattice, indices, values):
    """Set (discard previous) yaw rotation errors to lattice.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit [rad]
    """
    #processes arguments
    indices, values = _process_args_errors(indices, values)

    #set new values to first T1 and last T2
    for i in range(len(indices)):
        segs = indices[i]
        angx = -values[i]
        L    = sum([lattice[ii].length for ii in segs])
        #It is possible that there is a misalignment error, so:
        misx = (lattice[segs[0]].t_in[0] - lattice[segs[-1]].t_out[0])/2

        # correction of the path length
        old_angy = lattice[segs[0]].t_in[3]
        path = -(L/2)*(angx*angx + old_angy*old_angy)

        #Apply the errors only to the entrance of the first and exit of the last segment:
        lattice[segs[ 0]].t_in[0]  = -(L/2)*angx+misx
        lattice[segs[-1]].t_out[0] = -(L/2)*angx-misx
        lattice[segs[ 0]].t_in[1]  =  angx
        lattice[segs[-1]].t_out[1] = -angx
        lattice[segs[-1]].t_out[5] =  path


@_interactive
def add_error_rotation_yaw(lattice, indices, values):
    """Add (sum to previous) yaw rotation errors to lattice elements.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit: [rad]
    """

    #processes arguments
    indices, values = _process_args_errors(indices, values)

    #set new values to first T1 and last T2. Uses small angle approximation
    for i in range(len(indices)):
        segs = indices[i]
        angx  = -values[i]
        L    = sum([lattice[ii].length for ii in segs])

        # correction of the path length
        old_angx = lattice[segs[0]].t_in[1]
        path = -(L/2)*((angx+old_angx)*(angx+old_angx) - old_angx*old_angx)

        #Apply the errors only to the entrance of the first and exit of the last segment:
        lattice[segs[ 0]].t_in  += _numpy.array([-(L/2)*angx, angx,0,0,0,0])
        lattice[segs[-1]].t_out += _numpy.array([-(L/2)*angx,-angx,0,0,0,path])


@_interactive
def add_error_excitation_main(lattice, indices, values):
    """ Add excitation errors to magnets.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices. Unit: Relative value
    """
    #processes arguments
    indices, values = _process_args_errors(indices, values)

    for i in range(len(indices)):
        segs = indices[i]
        error = values[i]
        for j in range(len(segs)):
            idx = segs[j]
            if lattice[idx].angle != 0:
                rho = lattice[idx].length / lattice[idx].angle
                # read dipole pass method!
                lattice[idx].polynom_b[0] += error/rho
#               lattice[idx].polynom_a[1:] *= 1 + error
#               lattice[idx].polynom_b[1:] *= 1 + error
            else:
                lattice[idx].hkick *= 1 + error
                lattice[idx].vkick *= 1 + error
                lattice[idx].polynom_a *= 1 + error
                lattice[idx].polynom_b *= 1 + error


@_interactive
def add_error_excitation_kdip(lattice, indices, values):
    """ Add excitation errors to the quadrupole component of dipoles.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the errros. If the elements are segmented in the model
        and the same error is to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      values : may be a float or a (list, tuple, 1D numpy.ndarray) of floats
        with the same length as indices.
    """
    #processes arguments
    indices, values = _process_args_errors(indices, values)

    for i in range(len(indices)):
        segs = indices[i]
        for j in range(len(segs)):
            idx = segs[j]
            if lattice[idx].angle != 0:
                lattice[idx].polynom_b[1] *= 1 + values[i]
            else:
                raise TypeError('lattice[{0:d}] is not a Bending Magnet.'.format(idx))


@_interactive
def add_error_multipoles(lattice, indices, r0, main_monom, Bn_norm=None, An_norm=None):
    """ Add multipole errors to elements of lattice.

    INPUTS:
      lattice : accelerator model
      indices : (list, tuple, numpy.ndarray) of the indices of elements to
        appy the multipole errros. If the elements are segmented in the model
        and the same errors are to be applied to each segment, then it must be
        a (nested list,nested tuple, 2D numpy.ndarray), where each of its
        (elements, elements, first dimension) is a (list/tuple, tuple/list, 1D
        numpy.ndarray) of indices of the segments. Elements may have different
        number of segments.
      r0      : float whose meaning is the transverse horizontal position where
        the multipoles are normalized. Unit [meters];
      main_monom : may be an integer or (list, tuple, 1D numpy.ndarray) of integers
        whose meaning is the order of the main field strength compoment of each
        element. Positive values mean the main field component is normal and
        negative values mean they are skew. Examples:
          n= 1: dipole or horizontal corrector
          n=-1: vertical corrector
          n= 2: normal quadrupole
          n=-2: skew quadrupole  and so on
      Bn_norm : may be one normalized polynom to be applied to all elements or
        a list of normalized polynoms, one for each element. If the normalized
        polynoms for each element have the same sizes, it can also be a 2D
        numpy.ndarray where the first dimension has the same length as indices.
        By normalized polynom we mean a list, tuple or 1D numpy.ndarray whose
        (i+1)-th element is given by:
            Bn_norm[i] = DeltaB[i]/B  @ r0      with
            DeltaB[i] = PolB[i] * r0**i    and    B = Kn * r0**(n-1)
        where n is the absolute value of main_monom, Kn is the principal
        field strength component of the element and PolB is the quantity which
        will be applied to the element.
        The default value is None, which means the polynom_b of the elements
        will not be affected.
      An_norm : analogous of Bn_norm but for the polynom_a.

    """

    def add_polynom(elem, polynom, Pol_norm, n, KP):
        if Pol_norm is not None:
            if isinstance(Pol_norm,_numpy.ndarray):
                Pol = Pol_norm
            else:
                Pol = _numpy.array(Pol_norm)
            monoms = abs(n-1) - np.arange(Pol.shape[0])
            r0_i = r0**monoms
            newPol = KP*r0_i*Pol
            oldPol = getattr(elem,polynom)
            lenNewPol = len(newPol)
            lenOldPol = len(oldPol)
            if lenNewPol > lenOldPol:
                pol = newPol
                pol[:lenOldPol] += oldPol
            else:
                pol = oldPol
                pol[:lenNewPol] += newPol
            setattr(elem, polynom, pol)


    indices, *_ = _process_args_errors(indices, 0.0)

    if len(main_monom)==1:
        main_monom *= _numpy.ones(len(indices))
    if len(main_monom) != len(indices):
        raise IndexError('Length of main_monoms differs from length of indices.')

    #Extend the fields, if necessary to the number of elements in indices
    types = (int,float,_numpy.int64,_numpy.int32,_numpy.float64,_numpy.float32)
    if Bn_norm is None or isinstance(Bn_norm[0],types):
        Bn_norm = len(indices) * [Bn_norm]
    if An_norm is None or isinstance(An_norm[0],types):
        An_norm = len(indices) * [An_norm]
    if len(Bn_norm) != len(indices) or len(An_norm) != len(indices):
        raise IndexError('Length of polynoms differs from length of indices.')

    for i in range(len(indices)):
        segs = indices[i]
        n  = main_monom[i]
        for j in range(len(segs)):
            idx = segs[j]
            if abs(n)==1  and lattice[idx].angle != 0:
                if lattice[idx].length > 0 :
                    KP = lattice[idx].angle/lattice[idx].length
                else:
                    KP = lattice[idx].angle
            else:
                if n > 0:
                    KP = lattice[idx].polynom_b[n-1]
                else:
                    KP = lattice[idx].polynom_a[-n-1]
            add_polynom(lattice[idx],'polynom_b', Bn_norm[i], n, KP)
            add_polynom(lattice[idx],'polynom_a', An_norm[i], n, KP)


def _process_args_errors(indices, values):
    types = (int,_numpy.int64,_numpy.int32)
    if isinstance(indices,types):
        indices = [[indices]]
    elif len(indices) > 0 and isinstance(indices[0],types):
        indices = [[ind] for ind in indices]

    types = (int,float,_numpy.int64,_numpy.int32,_numpy.float64,_numpy.float32)
    if isinstance(values,types):
        values = len(indices) * [values]
    if len(values) != len(indices):
        raise IndexError('length of values differs from length of indices.')
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
