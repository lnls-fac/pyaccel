
import matplotlib.collections as _collections
import matplotlib.patches as _patches


_COLOURS = {
    'dipole': '#3b83bd',
    'quadrupole': '#f75e25',
    'sextupole': '#89ac76',
    'corrector': '#cccccc',
    'coil': '#641b1b'
}


class LatticeDrawer(object):

    def __init__(self, lattice, offset=0.0, height=1.0, print_edges=True,
            family_data=None, colours=None):
        self._offset = offset
        self._height = height

        if colours is None:
            colours = _COLOURS

        self._dipole_patches = []
        self._quadrupole_patches = []
        self._sextupole_patches = []
        self._bpm_patches = []
        self._corrector_patches = []

        pos = 0.0
        for element in lattice:
            self._create_element_patch(element, pos)
            pos += element.length

        ec = 'black'
        self.patch_collections = {
            'dipole': _collections.PatchCollection(
                self._dipole_patches,
                edgecolor=(ec if print_edges else colours['dipole']),
                facecolor=colours['dipole']
            ),
            'quadrupole': _collections.PatchCollection(
                self._quadrupole_patches,
                edgecolor=(ec if print_edges else colours['quadrupole']),
                facecolor=colours['quadrupole']
            ),
            'sextupole': _collections.PatchCollection(
                self._sextupole_patches,
                edgecolor=(ec if print_edges else colours['sextupole']),
                facecolor=colours['sextupole']
            ),
        }

    def _create_element_patch(self, element, pos):
        if element.pass_method == 'identity_pass':
            pass
        elif element.pass_method == 'drift_pass':
            pass
        elif element.angle != 0:
            r = self._get_rectangle(element, pos)
            self._dipole_patches.append(r)
        elif element.polynom_b[1] != 0:
            r = self._get_rectangle(element, pos)
            self._quadrupole_patches.append(r)
        elif element.polynom_b[2] != 0:
            r = self._get_rectangle(element, pos)
            self._sextupole_patches.append(r)
        else:
            pass

    def _get_rectangle(self, element, pos):
        corner = (pos, self._offset-self._height/2)
        rectangle = _patches.Rectangle(
            xy=corner,
            width=element.length,
            height=self._height,
        )
        return rectangle
