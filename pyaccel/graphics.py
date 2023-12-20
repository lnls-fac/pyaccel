"""."""

import matplotlib.pyplot as _plt
import matplotlib.lines as _lines
import matplotlib.collections as _collections
import matplotlib.patches as _patches

from .utils import interactive as _interactive
from .lattice import find_spos as _find_spos, find_indices as _find_indices, \
    get_attribute as _get_attribute
from .optics import calc_twiss as _calc_twiss


_COLOURS = {
    'dipole': '#3b83bd',
    'quadrupole': '#f75e25',
    'sextupole': '#89ac76',
    'septum': '#dd0000',
    'corrector': '#bbbbbb',
    'skew_quadupole': '#aa1b1b',
    'coil': '#444444',
    'bpm': '#444444',
    'vacuum_chamber': '#444444',
    'girder': '#EBF086',
    }


@_interactive
def plot_twiss(accelerator, twiss=None, plot_eta=True, add_lattice=True,
               offset=None, height=1.0, draw_edges=False, family_data=None,
               family_mapping=None, colours=None, selection=None,
               symmetry=None, gca=False, grid=False, title=None,
               show_label=False):
    """Plot Twiss parameters and draw lattice.

    Keyword arguments:
    accelerator -- Accelerator instance
    twiss -- Twiss parameters (first output from pyaccel.optics.calc_twiss)
    plot_eta -- Plot dispersion (default: True)
    add_lattice -- Add lattice drawing (default: True)
    For the other arguments, see draw_lattice documentation.

    Raises RuntimeError
    """
    if twiss is None:
        twiss, *_ = _calc_twiss(accelerator)

    spos = twiss.spos
    betax = twiss.betax
    betay = twiss.betay
    etax = twiss.etax

    if symmetry is not None:
        max_length = accelerator.length/symmetry
        for idx, _ in enumerate(accelerator):
            if spos[idx] >= max_length:
                accelerator = accelerator[:idx]
                spos = spos[:idx+1]
                betax = betax[:idx+1]
                betay = betay[:idx+1]
                etax = etax[:idx+1]
                break

    is_interactive = _plt.isinteractive()
    _plt.interactive = False

    if gca is True:
        fig = _plt.gcf()
        axis = _plt.gca()
    elif gca is False:
        fig, axis = _plt.subplots()
    else:
        axis = gca
        fig = axis.figure

    _plt.plot(spos, betax, label='$\\beta_x$', color='#085199')
    _plt.plot(spos, betay, label='$\\beta_y$', color='#990851')
    _plt.xlabel('s [m]')
    _plt.ylabel('$\\beta$ [m]')
    if grid:
        _plt.grid()
    if title:
        _plt.title(title)
    if add_lattice:
        _, y_max = _plt.ylim()

    if add_lattice:
        fig, axis = draw_lattice(
            accelerator, offset, height, draw_edges, family_data,
            family_mapping, colours, selection, gca=True,
            is_interactive=False, show_label=show_label)

    handles, labels = axis.get_legend_handles_labels()
    if plot_eta:
        eta_ax = axis.twinx()
        eta_colour = '#519908'
        eta_ax.plot(spos, 100*etax, label='$\\eta_x$', color=eta_colour)
        eta_ax.set_ylabel('$\\eta_x$ [cm]')
        # eta_ax.spines['right'].set_color(eta_colour)
        # eta_ax.tick_params(axis='y', colors=eta_colour)
        eta_y_min, eta_y_max = eta_ax.get_ylim()
        y_min, _ = axis.get_ylim()
        if show_label:
            eta_ax.set_ylim(
                bottom=eta_y_min+(y_min/y_max)*(eta_y_max-eta_y_min),
                top=eta_y_max)
        else:
            eta_ax.set_ylim(
                bottom=eta_y_min+y_min, top=eta_y_max-y_min)

        eta_handles, eta_labels = eta_ax.get_legend_handles_labels()
        handles += eta_handles
        labels += eta_labels
        axis.legend(handles, labels)

        _plt.sca(axis)
    else:
        axis.legend(handles, labels)

    if not gca:
        _plt.xlim((spos[0]-0.25, spos[-1]+0.25))

    if is_interactive:
        if add_lattice:
            y_min, _ = _plt.ylim()
            _plt.ylim(y_min, y_max)
        _plt.interactive = True
        _plt.draw()
        _plt.show()

    return fig, axis


@_interactive
def plot_vchamber(accelerator, add_lattice=True,
                  offset=None, height=5.0, draw_edges=False, family_data=None,
                  family_mapping=None, colours=None, selection=None,
                  grid=False, show_label=False):
    """."""
    def plot(umax, umin, title, ylabel, color):
        spos = _find_spos(accelerator)
        xmin, xmax = min(spos), max(spos)
        ymin, ymax = min(min(umax), min(umin)), max(max(umax), max(umin))
        difx, dify = xmax - xmin, ymax - ymin
        xmin, xmax = xmin - 0.05 * difx, xmax + 0.05 * difx
        ymin, ymax = ymin - 0.05 * dify, ymax + 0.05 * dify

        upper = []
        for i in range(len(spos)-1):
            x1_, y1_ = spos[i], umax[i]
            x2_, y2_ = spos[i+1], umax[i]
            upper.append([(x1_, y1_), (x2_, y2_)])
            x1_, y1_ = spos[i+1], umax[i]
            x2_, y2_ = spos[i+1], umax[i+1]
            upper.append([(x1_, y1_), (x2_, y2_)])
        center = [[(xmin, 0), (xmax, 0)], ]
        lower = []
        for i in range(len(spos)-1):
            x1_, y1_ = spos[i], umin[i]
            x2_, y2_ = spos[i+1], umin[i]
            lower.append([(x1_, y1_), (x2_, y2_)])
            x1_, y1_ = spos[i+1], umin[i]
            x2_, y2_ = spos[i+1], umin[i+1]
            lower.append([(x1_, y1_), (x2_, y2_)])

        fig, axis = _plt.subplots()
        _ = fig
        lines_upper = _collections.LineCollection(
            upper, color=color, linewidths=1)
        lines_lower = _collections.LineCollection(
            lower, color=color, linewidths=1)
        lines_center = _collections.LineCollection(
            center, color=color, linewidths=1, linestyles='dashed')
        axis.add_collection(lines_upper)
        axis.add_collection(lines_lower)
        axis.add_collection(lines_center)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_xlabel('pos [m]')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if grid:
            axis.grid()

        if add_lattice:
            fig, axis = draw_lattice(
                accelerator, offset, height, draw_edges, family_data,
                family_mapping, colours, selection, gca=True,
                is_interactive=False, show_label=show_label)

        _plt.show()

    # horizontal
    hmax = 1e3 * _get_attribute(accelerator, 'hmax')  # [mm]
    hmin = 1e3 * _get_attribute(accelerator, 'hmin')  # [mm]
    plot(hmax, hmin, 'Vacuum Chamber Horizontal Limits', 'X [mm]', 'blue')

    # vertical
    vmax = 1e3 * _get_attribute(accelerator, 'vmax')  # [mm]
    vmin = 1e3 * _get_attribute(accelerator, 'vmin')  # [mm]
    plot(vmax, vmin, 'Vacuum Chamber Vertical Limits', 'Y [mm]', 'red')


@_interactive
def draw_lattice(lattice, offset=None, height=1.0, draw_edges=False,
                 draw_girders=False, family_data=None, family_mapping=None,
                 colours=None, selection=None, symmetry=None, gca=False,
                 is_interactive=None, show_label=False):
    """Draw lattice elements along longitudinal position.

    Keyword arguments:
    lattice -- Accelerator or Element list
    offset -- Element center vertical offset
    height -- Element height
    draw_edges -- If True, draw element edges in black
    draw_girders -- If True, draw girders
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
        'pulsed_magnets' (equivalent to 'septum')
    symmetry -- lattice symmetry (draw only one period)
    gca -- use current pyplot Axes instance (default: False)
    is_interactive -- pyplot interactive status
    show_label -- If True, show labels of the elements

    Returns:
    fig -- matplotlib Figure object
    axis -- matplotlib AxesSubplot object

    Raises:
    RuntimeError

    """
    if selection is None:
        selection = [
            'dipole',
            'quadrupole',
            'sextupole',
            'septum',
            'fast_corrector_core',
            'fast_corrector_coil',
            'slow_corrector_core',
            'slow_corrector_coil',
            'skew_quadupole_core',
            'skew_quadupole_coil',
            'bpm']
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
        if 'pulsed_magnets' in selection:
            selection.remove('pulsed_magnets')
            selection.append('septum')

    if draw_girders:
        selection.append('girder')

    if is_interactive is None:
        is_interactive = _plt.isinteractive()
    _plt.interactive = False

    if gca is True:
        fig = _plt.gcf()
        axis = _plt.gca()
    elif gca is False:
        fig, axis = _plt.subplots()
    else:
        axis = gca
        fig = axis.figure

    if offset is None:
        offset = 0.0

    if symmetry is not None:
        max_length = lattice.length/symmetry
        spos = 0
        for i, ele in enumerate(lattice):
            spos += ele.length
            if spos >= max_length:
                lattice = lattice[:i]
                break

    line = _lines.Line2D(
        [0, lattice.length], [offset, offset],
        color=_COLOURS['vacuum_chamber'], linewidth=1)
    line.set_zorder(0)
    axis.add_line(line)

    drawer = _LatticeDrawer(
        lattice, offset, height, draw_edges, draw_girders, family_data,
        family_mapping, colours, show_label)

    if not gca:
        axis.set_xlim(0, lattice.length)
        axis.set_ylim(offset-height, offset+19*height)

    for s in selection:
        axis.add_collection(drawer.patch_collections[s])

    if show_label:
        for l in drawer.patch_labels:
            axis.text(x=l[0], y=l[1], s=l[2], rotation='vertical', fontsize=8)

    if is_interactive:
        _plt.interactive = True
        _plt.draw()
        _plt.show()

    return fig, axis


class _LatticeDrawer(object):
    """."""

    def __init__(
            self, lattice, offset, height, draw_edges, draw_girders,
            family_data, family_mapping, colours, show_label):
        """."""
        self._show_label = show_label
        self._bpm_length = 0.10
        self._coil_length = 0.15

        self._offset = offset
        self._height = height

        self._fast_corrector_height = 0.42*height
        self._septum_height = 0.50*height
        self._coil_height = 0.10*height
        self._bpm_height = 0.10*height

        if colours is None:
            colours = _COLOURS

        self._dipole_patches = []
        self._quadrupole_patches = []
        self._sextupole_patches = []
        self._septum_patches = []
        self._fast_corrector_core_patches = []
        self._fast_corrector_coil_patches = []
        self._slow_corrector_core_patches = []
        self._slow_corrector_coil_patches = []
        self._skew_quadrupole_core_patches = []
        self._skew_quadrupole_coil_patches = []
        self._bpm_patches = []
        self._girder_patches = []
        if show_label:
            self.patch_labels = []

        pos = _find_spos(lattice)

        if family_data is None:
            # Guess element type
            for i in range(len(lattice)):
                self._create_element_patch(lattice[i], pos[i])
        else:
            # family_data is not None; we need a family_mapping to proceed
            if family_mapping is None:
                raise RuntimeError('missing family_mapping argument')

            for key in family_mapping.keys():
                indices = []
                for i, v in enumerate(family_data[key]['index']):
                    indices.extend(v)
                    if not show_label:
                        continue
                    inst = family_data[key]['instance'][i]
                    if key == 'Scrn':
                        tup = [
                            pos[v[0]], self._offset+1.25*self._height,
                            key+inst]
                    else:
                        tup = [pos[v[0]], self._offset-self._height, key+inst]
                    if tup not in self.patch_labels:
                        self.patch_labels.append(tup)
                        if key not in ('ICT', 'FCT', 'BPM'):
                            self.patch_labels.append(tup)

                et = family_mapping[key]
                for i in indices:
                    if i > len(lattice):
                        break
                    self._create_element_patch(lattice[i], pos[i], et)

        edgec = 'black'
        self.patch_collections = {
            'dipole': _collections.PatchCollection(
                self._dipole_patches,
                edgecolor=(edgec if draw_edges else colours['dipole']),
                facecolor=colours['dipole'],
                zorder=2),
            'quadrupole': _collections.PatchCollection(
                self._quadrupole_patches,
                edgecolor=(edgec if draw_edges else colours['quadrupole']),
                facecolor=colours['quadrupole'],
                zorder=2),
            'sextupole': _collections.PatchCollection(
                self._sextupole_patches,
                edgecolor=(edgec if draw_edges else colours['sextupole']),
                facecolor=colours['sextupole'],
                zorder=2),
            'septum': _collections.PatchCollection(
                self._septum_patches,
                edgecolor=(edgec if draw_edges else colours['septum']),
                facecolor=colours['septum'],
                zorder=2),
            'fast_corrector_core': _collections.PatchCollection(
                self._fast_corrector_core_patches,
                edgecolor=(edgec if draw_edges else colours['corrector']),
                facecolor=colours['corrector'],
                zorder=2),
            'fast_corrector_coil': _collections.PatchCollection(
                self._fast_corrector_coil_patches,
                edgecolor=(edgec if draw_edges else colours['coil']),
                facecolor=colours['coil'],
                zorder=3),
            'slow_corrector_core': _collections.PatchCollection(
                self._slow_corrector_core_patches,
                edgecolor=(edgec if draw_edges else colours['corrector']),
                facecolor=colours['corrector'],
                zorder=2),
            'slow_corrector_coil': _collections.PatchCollection(
                self._slow_corrector_coil_patches,
                edgecolor=(edgec if draw_edges else colours['coil']),
                facecolor=colours['coil'],
                zorder=3),
            'skew_quadupole_core': _collections.PatchCollection(
                self._skew_quadrupole_core_patches,
                edgecolor=(edgec if draw_edges else colours['skew_quadupole']),
                facecolor=colours['skew_quadupole'],
                zorder=2),
            'skew_quadupole_coil': _collections.PatchCollection(
                self._skew_quadrupole_coil_patches,
                edgecolor=(edgec if draw_edges else colours['coil']),
                facecolor=colours['coil'],
                zorder=3),
            'bpm': _collections.PatchCollection(
                self._bpm_patches,
                edgecolor=(edgec if draw_edges else colours['bpm']),
                facecolor=colours['bpm'],
                zorder=2),
            }

        if draw_girders:
            girs = _find_indices(lattice, 'fam_name', 'girder')
            if girs:
                for ini, fin in zip(girs[::2], girs[1::2]):
                    self._girder_patches.append(self._get_girder(
                        pos[ini], pos[fin]))

                self.patch_collections['girder'] = \
                    _collections.PatchCollection(
                        self._girder_patches,
                        edgecolor=(edgec if draw_edges else colours['girder']),
                        facecolor=colours['girder'],
                        zorder=-2)

    def _create_element_patch(self, element, pos, element_type=None):
        if element_type is None:
            element_type = self._guess_element_type(element)

        if element_type in ('marker', 'drift'):
            pass
        elif element_type == 'pulsed_magnet':
            r = self._get_septum_core(element, pos)
            self._septum_patches.append(r)
        elif element_type == 'dipole':
            r = self._get_magnet_core(element, pos)
            self._dipole_patches.append(r)
        elif element_type == 'quadrupole':
            r = self._get_magnet_core(element, pos)
            self._quadrupole_patches.append(r)
        elif element_type == 'sextupole':
            r = self._get_magnet_core(element, pos)
            self._sextupole_patches.append(r)
        elif element_type == 'fast_horizontal_corrector':
            r1 = self._get_corrector_core(element, pos)
            self._fast_corrector_core_patches.append(r1)
            r2 = self._get_fast_horizontal_corrector_coil(element, pos)
            self._fast_corrector_coil_patches.append(r2)
        elif element_type == 'fast_vertical_corrector':
            r1 = self._get_corrector_core(element, pos)
            self._fast_corrector_core_patches.append(r1)
            r2 = self._get_fast_vertical_corrector_coil(element, pos)
            self._fast_corrector_coil_patches.append(r2)
        elif element_type == 'fast_corrector':
            r1 = self._get_corrector_core(element, pos)
            self._fast_corrector_core_patches.append(r1)
            r2 = self._get_fast_horizontal_corrector_coil(element, pos)
            r3 = self._get_fast_vertical_corrector_coil(element, pos)
            self._fast_corrector_coil_patches.extend([r2, r3])
        elif element_type in ('slow_horizontal_corrector',
                              'horizontal_corrector'):
            r = self._get_slow_horizontal_corrector_coil(element, pos)
            self._slow_corrector_coil_patches.append(r)
        elif element_type in ('slow_vertical_corrector', 'vertical_corrector'):
            r = self._get_slow_vertical_corrector_coil(element, pos)
            self._slow_corrector_coil_patches.append(r)
        elif element_type == 'skew_quadrupole':
            r = self._get_skew_quadrupole(element, pos)
            self._skew_quadrupole_coil_patches.append(r)
        elif element_type == 'bpm':
            r = self._get_bpm(element, pos)
            self._bpm_patches.append(r)
        else:
            pass

    def _guess_element_type(self, element):
        if element.fam_name in ('bpm', 'BPM'):
            return 'bpm'
        elif element.pass_method == 'identity_pass':
            return 'marker'
        elif element.pass_method == 'drift_pass':
            return 'drift'
        elif element.fam_name in ('EjeSF', 'EjeSG', 'InjSF', 'InjSG'):
            return 'pulsed_magnet'
        elif element.angle != 0:
            return 'dipole'
        elif element.polynom_b[1] != 0:
            return 'quadrupole'
        elif element.polynom_b[2] != 0:
            return 'sextupole'
        elif element.fam_name in ('CH', 'horizontal_corrector'):
            return 'slow_horizontal_corrector'
        elif element.fam_name in ('CV', 'vertical_corrector'):
            return 'slow_vertical_corrector'
        else:
            return 'unknown'

    def _get_magnet_core(self, element, pos):
        corner = (pos, self._offset-self._height/2)
        return _patches.Rectangle(
            xy=corner, width=element.length, height=self._height)

    def _get_septum_core(self, element, pos):
        corner = (pos, self._offset-self._septum_height/2)
        return _patches.Rectangle(
            xy=corner, width=element.length, height=self._septum_height)

    def _get_corrector_core(self, element, pos):
        corner = (pos, self._offset-self._fast_corrector_height/2)
        return _patches.Rectangle(
            xy=corner, width=element.length,
            height=self._fast_corrector_height)

    def _get_slow_horizontal_corrector_coil(self, element, pos):
        corner = (
            pos-self._coil_length/2,
            self._offset+self._height/2-self._coil_height)
        return self._get_coil(element, corner)

    def _get_slow_vertical_corrector_coil(self, element, pos):
        corner = (pos-self._coil_length/2, self._offset-self._height/2)
        return self._get_coil(element, corner)

    def _get_fast_horizontal_corrector_coil(self, element, pos):
        y = self._offset + self._fast_corrector_height/2 - self._coil_height
        corner = (pos, y)
        return self._get_coil(element, corner)

    def _get_fast_vertical_corrector_coil(self, element, pos):
        y = self._offset - self._fast_corrector_height/2
        corner = (pos, y)
        return self._get_coil(element, corner)

    def _get_skew_quadrupole(self, element, pos):
        corner = (pos, self._offset-self._coil_height/2)
        return self._get_coil(element, corner)

    def _get_coil(self, element, corner):
        _ = element
        return _patches.Rectangle(
            xy=corner, width=self._coil_length, height=self._coil_height)

    def _get_bpm(self, element, pos):
        _ = element
        corner = (pos-self._bpm_length/2, self._offset-self._height/20)
        return _patches.Rectangle(
            xy=corner, width=self._bpm_length, height=self._height/10)

    def _get_girder(self, posi, posf):
        corner = (posi-5e-3, self._offset - self._height*6/10)
        return _patches.Rectangle(
            xy=corner, width=posf-posi + 1e-2, height=self._height*12/10)
