
import matplotlib.pyplot as _pyplot
import matplotlib.lines as _lines
import matplotlib.collections as _collections
import matplotlib.patches as _patches
import pyaccel as _pyaccel
from pyaccel.utils import interactive as _interactive


_COLOURS = {
    'dipole': '#3b83bd',
    'quadrupole': '#f75e25',
    'sextupole': '#89ac76',
    'corrector': '#bbbbbb',
    'skew_quadupole': '#aa1b1b',
    'coil': '#444444',
    'bpm': '#444444',
    'vacuum_chamber': '#444444'
}


@_interactive
def plottwiss(accelerator, twiss=None, plot_eta=True, draw_lattice=True,
        offset=None, height=1.0, draw_edges=False, family_data=None,
        family_mapping=None, colours=None, selection=None, symmetry=None,
        gca=False):
    """Plot Twiss parameters and draw lattice.

    Keyword arguments:
    accelerator -- Accelerator instance
    twiss -- Twiss parameters (output from pyaccel.optics.calctwiss)
    plot_eta -- Plot dispersion (default: True)
    draw_lattice -- Add lattice drawing (default: True)
    For the other arguments, see drawlattice documentation.

    Raises RuntimeError"""
    if twiss is None:
        twiss = _pyaccel.optics.calctwiss(accelerator)

    spos, betax, betay, etax = _pyaccel.optics.gettwiss(
        twiss[0],
        ('spos', 'betax', 'betay', 'etax')
    )

    if symmetry is not None:
        max_length = accelerator.length/symmetry
        s = 0
        for i in range(len(accelerator)):
            s += accelerator[i].length
            if s >= max_length:
                break

        accelerator = accelerator[:i]
        spos = spos[:i+1]
        betax = betax[:i+1]
        betay = betay[:i+1]
        etax = etax[:i+1]

    is_interactive = _pyplot.isinteractive()
    _pyplot.interactive(False)

    if gca:
        fig = _pyplot.gcf()
        ax = _pyplot.gca()
    else:
        fig, ax = _pyplot.subplots()

    _pyplot.plot(spos, betax, spos, betay)
    _pyplot.xlabel('s [m]')
    _pyplot.ylabel('$\\beta$ [m]')
    if draw_lattice:
        _, y_max = _pyplot.ylim()

    legend = ax.legend(('$\\beta_x$', '$\\beta_y$'))

    if plot_eta:
        eta_ax = ax.twinx()
        eta_colour = 'red'
        eta_ax.plot(spos, etax, color=eta_colour)
        eta_ax.set_ylabel('$\\eta_x$ [m]', color=eta_colour)
        eta_ax.spines['right'].set_color(eta_colour)
        eta_ax.tick_params(axis='y', colors=eta_colour)
        eta_ax.add_artist(legend)
        ax.legend = None
        _pyplot.sca(ax)

    if draw_lattice:
        fig, ax = drawlattice(accelerator, offset, height, draw_edges,
            family_data, family_mapping, colours, selection, gca=True)

    if is_interactive:
        if draw_lattice:
            y_min, _ = _pyplot.ylim()
            _pyplot.ylim(y_min, y_max)
        _pyplot.interactive(True)
        _pyplot.draw()
        _pyplot.show()

    return fig, ax


@_interactive
def drawlattice(lattice, offset=None, height=1.0, draw_edges=False,
        family_data=None, family_mapping=None, colours=None, selection=None,
        symmetry=None, gca=False):
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
    symmetry -- lattice symmetry (draw only one period)
    gca -- use current pyplot Axes instance (default: False)

    Returns:
    fig -- matplotlib Figure object
    ax -- matplotlib AxesSubplot object

    Raises RuntimeError
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

    if symmetry is not None:
        max_length = lattice.length/symmetry
        s = 0
        for i in range(len(lattice)):
            s += lattice[i].length
            if s >= max_length:
                break

        lattice = lattice[:i]


    line = _lines.Line2D([0, lattice.length], [offset, offset],
        color=_COLOURS['vacuum_chamber'], linewidth=1)
    line.set_zorder(0)
    ax.add_line(line)

    drawer = _LatticeDrawer(lattice, offset, height, draw_edges, family_data,
        family_mapping, colours)

    ax.set_xlim(0, lattice.length)
    ax.set_ylim(offset-height, offset+19*height)

    for s in selection:
        ax.add_collection(drawer.patch_collections[s])

    if is_interactive:
        if gca:
            _pyplot.ylim(offset-height, y_max)
        _pyplot.interactive(True)
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

        pos = _pyaccel.lattice.findspos(lattice)

        if family_data is None:
            # Guess element type; draw only magnetic lattice
            for i in range(len(lattice)):
                self._create_element_patch(lattice[i], pos[i])
        else:
            # family_data is not None; we need a family_mapping to proceed
            if family_mapping is None:
                raise RuntimeError('missing family_mapping argument')

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
                    if i > len(lattice):
                        break
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
            r1 = self._get_magnet(element, pos)
            self._fast_corrector_core_patches.append(r1)
            r2 = self._get_horizontal_corrector_coil(element, pos)
            self._fast_corrector_coil_patches.append(r2)
        elif element_type == 'fast_vertical_corrector':
            r1 = self._get_magnet(element, pos)
            self._fast_corrector_core_patches.append(r1)
            r2 = self._get_vertical_corrector_coil(element, pos)
            self._fast_corrector_coil_patches.append(r2)
        elif element_type == 'fast_corrector':
            r1 = self._get_magnet(element, pos)
            self._fast_corrector_core_patches.append(r1)
            r2 = self._get_horizontal_corrector_coil(element, pos)
            r3 = self._get_vertical_corrector_coil(element, pos)
            self._fast_corrector_coil_patches.extend([r2, r3])
        elif element_type == 'slow_horizontal_corrector':
            r = self._get_horizontal_corrector_coil(element, pos)
            self._slow_corrector_coil_patches.append(r)
        elif element_type == 'slow_vertical_corrector':
            r = self._get_vertical_corrector_coil(element, pos)
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

    def _get_horizontal_corrector_coil(self, element, pos):
        w = element.length + 2*self._coil_length
        h = self._height/10
        corner = (pos-self._coil_length, self._offset+4*self._height/10)
        r = _patches.Rectangle(xy=corner, width=w, height=h)
        return r

    def _get_vertical_corrector_coil(self, element, pos):
        w = element.length + 2*self._coil_length
        h = self._height/10
        corner = (pos-self._coil_length, self._offset-5*self._height/10)
        r = _patches.Rectangle(xy=corner, width=w, height=h)
        return r

    def _get_skew_quadrupole(self, element, pos):
        w = element.length + 2*self._coil_length
        h = self._height/10
        corner = (pos-self._coil_length, self._offset+-1*self._height/20)
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
