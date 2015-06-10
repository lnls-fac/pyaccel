
import math as _math
import numpy as _numpy
import matplotlib.pyplot as _pyplot
import matplotlib.lines as _lines
import matplotlib.collections as _collections
import matplotlib.patches as _patches
import mathphys as _mp
import trackcpp as _trackcpp
import pyaccel as _pyaccel
from pyaccel.utils import interactive as _interactive


_COLOURS = {
    'dipole': '#3b83bd',
    'quadrupole': '#f75e25',
    'sextupole': '#89ac76',
    'corrector': '#bbbbbb',
    'skew_quadupole': '#aa1b1b',
    'coil': '#641b1b',
    'bpm': '#111111'
}


class LatticeError(Exception):
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
def get_error_misalignment_x(lattice, indices):
    """Set horizontal misalignment errors to lattice"""

    ''' processes arguments '''
    indices, *_ = _process_args_errors(indices, 0.0)

    ''' loops over elements and gets error from T_IN '''
    values = []
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            index = indices[i,j]
            values.append(-lattice[index].t_in[0])
    if len(values) == 1:
        return values[0]
    else:
        return values

@_interactive
def set_error_misalignment_x(lattice, indices, values):
    """Set horizontal misalignment errors to lattice"""

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(indices.shape[0]):
        error = values[i]
        for j in range(indices.shape[1]):
            index = indices[i,j]
            lattice[index].t_in[0]  =  -values[i]
            lattice[index].t_out[0] =  +values[i]

    return lattice

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
def get_error_misalignment_y(lattice, indices):
    """Set horizontal misalignment errors to lattice"""

    ''' processes arguments '''
    indices, *_ = _process_args_errors(indices, 0.0)

    ''' loops over elements and gets error from T_IN '''
    values = []
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            index = indices[i,j]
            values.append(-lattice[index].t_in[2])
    if len(values) == 1:
        return values[0]
    else:
        return values

@_interactive
def set_error_misalignment_y(lattice, indices, values):
    """Set vertical misalignment errors to lattice"""

    ''' processes arguments '''
    indices, values = _process_args_errors(indices, values)

    ''' loops over elements and sets its T1 and T2 fields '''
    for i in range(indices.shape[0]):
        error = values[i]
        for j in range(indices.shape[1]):
            index = indices[i,j]
            lattice[index].t_in[2]  =  -values[i]
            lattice[index].t_out[2] =  +values[i]

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


@_interactive
def drawlattice(lattice, offset=None, height=1.0, draw_edges=False,
        family_data=None, family_mapping=None, colours=None, selection=None,
        gca=False):
    """Draw lattice elements along longitudinal position

    Keyword arguments:
    lattice -- Accelerator or Element list
    offset -- Element center vertical offset
    height -- Element height
    draw_edges -- If True, draw element edges in black
    family_data -- dict with family data; if supplied, family_mapping must also
        be passed
    family_mapping -- dict with mapping from family names to element types
    colours -- dict with element colours
    selection -- list or tuple of strings with element selection to be drawn
        (default: all); options are:
        'dipole'
        'quadrupole'
        'sextupole'
        'fast_corrector'
        'slow_corrector'
        'skew_quadrupole'
        'bpm'
        'magnets' (equivalent to 'dipole', 'quadrupole' and 'sextupole')
    gca -- use current pyplot Axes instance (default: False)

    Returns:
    fig -- matplotlib Figure object
    ax -- matplotlib AxesSubplot object

    Raises LatticeError
    """
    if selection is None:
        selection = [
            'dipole',
            'quadrupole',
            'sextupole',
            'fast_corrector_core',
            'fast_corrector_coil',
            'slow_corrector_core',
            'slow_corrector_coil',
            'skew_quadupole_core',
            'skew_quadupole_coil',
            'bpm'
        ]
    else:
        if 'slow_corrector' in selection:
            selection.remove('slow_corrector')
            selection.append('slow_corrector_core')
            selection.append('slow_corrector_coil')
        if 'fast_corrector' in selection:
            selection.remove('fast_corrector')
            selection.append('fast_corrector_core')
            selection.append('fast_corrector_coil')
        if 'skew_quadupole' in selection:
            selection.remove('skew_quadupole')
            selection.append('skew_quadupole_core')
            selection.append('skew_quadupole_coil')
        if 'magnets' in selection:
            selection.remove('magnets')
            selection.append('dipole')
            selection.append('quadrupole')
            selection.append('sextupole')

    is_interactive = _pyplot.isinteractive()
    _pyplot.interactive(False)

    if gca:
        fig = _pyplot.gcf()
        ax = _pyplot.gca()
        if offset is None:
            y_min, y_max = _pyplot.ylim()
            offset = y_min - height
    else:
        fig, ax = _pyplot.subplots()
        if offset is None:
            offset = 0.0

    line = _lines.Line2D([0, lattice.length], [offset, offset],
        color='#444444', linewidth=1)
    line.set_zorder(0)
    ax.add_line(line)

    drawer = _LatticeDrawer(lattice, offset, height, draw_edges, family_data,
        family_mapping, colours)

    ax.set_xlim(0, lattice.length)
    ax.set_ylim(offset-height, offset+19*height)

    for s in selection:
        ax.add_collection(drawer.patch_collections[s])

    if not is_interactive:
        return fig, ax
    else:
        _pyplot.interactive(True)
        _pyplot.ylim(offset-height, y_max)
        _pyplot.draw()
        _pyplot.show()
        return fig, ax


class _LatticeDrawer(object):

    def __init__(self, lattice, offset, height, draw_edges, family_data,
            family_mapping, colours):
        self._coil_length = 0.0
        self._bpm_length = 0.1

        self._offset = offset
        self._height = height

        if colours is None:
            colours = _COLOURS

        self._dipole_patches = []
        self._quadrupole_patches = []
        self._sextupole_patches = []
        self._fast_corrector_core_patches = []
        self._fast_corrector_coil_patches = []
        self._slow_corrector_core_patches = []
        self._slow_corrector_coil_patches = []
        self._skew_quadrupole_core_patches = []
        self._skew_quadrupole_coil_patches = []
        self._bpm_patches = []

        pos = findspos(lattice)

        if family_data is None:
            # Guess element type; draw only magnetic lattice
            for i in range(len(lattice)):
                self._create_element_patch(lattice[i], pos[i])
        else:
            # family_data is not None; we need a family_mapping to proceed
            if family_mapping is None:
                raise LatticeError('missing family_mapping argument')

            for key in family_mapping.keys():
                et = family_mapping[key]

                # Flatten index list for segmented elements, if necessary
                nr_segs = family_data[key].get('nr_segs', 1)
                if nr_segs > 1:
                    indices = []
                    for v in family_data[key]['index']:
                        for j in v:
                            indices.append(j)
                else:
                    indices = family_data[key]['index']

                for i in indices:
                    self._create_element_patch(lattice[i], pos[i], et)

        ec = 'black'
        self.patch_collections = {
            'dipole': _collections.PatchCollection(
                self._dipole_patches,
                edgecolor=(ec if draw_edges else colours['dipole']),
                facecolor=colours['dipole'],
                zorder=2,
            ),
            'quadrupole': _collections.PatchCollection(
                self._quadrupole_patches,
                edgecolor=(ec if draw_edges else colours['quadrupole']),
                facecolor=colours['quadrupole'],
                zorder=2,
            ),
            'sextupole': _collections.PatchCollection(
                self._sextupole_patches,
                edgecolor=(ec if draw_edges else colours['sextupole']),
                facecolor=colours['sextupole'],
                zorder=2,
            ),
            'fast_corrector_core': _collections.PatchCollection(
                self._fast_corrector_core_patches,
                edgecolor=(ec if draw_edges else colours['corrector']),
                facecolor=colours['corrector'],
                zorder=2,
            ),
            'fast_corrector_coil': _collections.PatchCollection(
                self._fast_corrector_coil_patches,
                edgecolor=(ec if draw_edges else colours['coil']),
                facecolor=colours['coil'],
                zorder=3,
            ),
            'slow_corrector_core': _collections.PatchCollection(
                self._slow_corrector_core_patches,
                edgecolor=(ec if draw_edges else colours['corrector']),
                facecolor=colours['corrector'],
                zorder=2,
            ),
            'slow_corrector_coil': _collections.PatchCollection(
                self._slow_corrector_coil_patches,
                edgecolor=(ec if draw_edges else colours['coil']),
                facecolor=colours['coil'],
                zorder=3,
            ),
            'skew_quadupole_core': _collections.PatchCollection(
                self._skew_quadrupole_core_patches,
                edgecolor=(ec if draw_edges else colours['skew_quadupole']),
                facecolor=colours['skew_quadupole'],
                zorder=2,
            ),
            'skew_quadupole_coil': _collections.PatchCollection(
                self._skew_quadrupole_coil_patches,
                edgecolor=(ec if draw_edges else colours['coil']),
                facecolor=colours['coil'],
                zorder=3,
            ),
            'bpm': _collections.PatchCollection(
                self._bpm_patches,
                edgecolor=(ec if draw_edges else colours['bpm']),
                facecolor=colours['bpm'],
                zorder=2,
            ),
        }

    def _create_element_patch(self, element, pos, element_type=None):
        if element_type is None:
            element_type = self._guess_element_type(element)

        if element_type in ('marker', 'drift'):
            pass
        elif element_type == 'dipole':
            r = self._get_magnet(element, pos)
            self._dipole_patches.append(r)
        elif element_type == 'quadrupole':
            r = self._get_magnet(element, pos)
            self._quadrupole_patches.append(r)
        elif element_type == 'sextupole':
            r = self._get_magnet(element, pos)
            self._sextupole_patches.append(r)
        elif element_type == 'fast_horizontal_corrector':
            pass
        elif element_type == 'fast_vertical_corrector':
            pass
        elif element_type == 'fast_corrector':
            pass
        elif element_type == 'slow_horizontal_corrector':
            r1, r2 = self._get_slow_horizontal_corrector(element, pos)
            self._slow_corrector_coil_patches.extend([r1, r2])
        elif element_type == 'slow_vertical_corrector':
            r1, r2 = self._get_slow_vertical_corrector(element, pos)
            self._slow_corrector_coil_patches.extend([r1, r2])
        elif element_type == 'skew_quadrupole':
            r = self._get_skew_quadrupole(element, pos)
            self._skew_quadrupole_coil_patches.append(r)
        elif element_type == 'bpm':
            r = self._get_bpm(element, pos)
            self._bpm_patches.append(r)
        else:
            pass

    def _guess_element_type(self, element):
        if element.pass_method == 'identity_pass':
            return 'marker'
        elif element.pass_method == 'drift_pass':
            return 'drift'
        elif element.angle != 0:
            return 'dipole'
        elif element.polynom_b[1] != 0:
            return 'quadrupole'
        elif element.polynom_b[2] != 0:
            return 'sextupole'
        else:
            return 'unknown'

    def _get_magnet(self, element, pos):
        corner = (pos, self._offset-self._height/2)
        r = _patches.Rectangle(
            xy=corner,
            width=element.length,
            height=self._height,
        )
        return r

    def _get_slow_horizontal_corrector(self, element, pos):
        w = element.length + 2*self._coil_length
        h = self._height/10
        c1 = (pos-self._coil_length, self._offset+4*self._height/10)
        c2 = (pos-self._coil_length, self._offset-5*self._height/10)
        r1 = _patches.Rectangle(xy=c1, width=w, height=h)
        r2 = _patches.Rectangle(xy=c2, width=w, height=h)
        return r1, r2

    def _get_slow_vertical_corrector(self, element, pos):
        w = element.length + 2*self._coil_length
        h = self._height/10
        c1 = (pos-self._coil_length, self._offset+2*self._height/10)
        c2 = (pos-self._coil_length, self._offset-3*self._height/10)
        r1 = _patches.Rectangle(xy=c1, width=w, height=h)
        r2 = _patches.Rectangle(xy=c2, width=w, height=h)
        return r1, r2

    def _get_skew_quadrupole(self, element, pos):
        w = element.length + 2*self._coil_length
        h = 2*self._height/10
        corner = (pos-self._coil_length, self._offset+-1*self._height/10)
        r = _patches.Rectangle(xy=corner, width=w, height=h)
        return r

    def _get_bpm(self, element, pos):
        corner = (pos-self._bpm_length/2, self._offset-self._height/20)
        r = _patches.Rectangle(
            xy=corner,
            width=self._bpm_length,
            height=self._height/10,
        )
        return r
