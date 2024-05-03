
# -*- coding: utf-8 -*-

# import ctypes as _ctypes
import warnings as _warnings
import numpy as _numpy

from .utils import interactive as _interactive

from mymodule import marker, bpm, drift, matrix, hcorrector, vcorrector, \
    corrector, rbend, quadrupole, sextupole, rfcavity, kickmap, \
    Element as _Element

@_interactive
class Kicktable:
    """."""

    kicktable_list = _trackcpp.cvar.kicktable_list  # trackcpp vector with all Kicktables in use.

    def __init__(self, **kwargs):
        """."""
        # get kicktable filename
        if 'kicktable' in kwargs:
            filename = kwargs['kicktable'].filename
        elif 'filename' in kwargs:
            filename = kwargs['filename']
        else:
            raise NotImplementedError('Invalid Kicktable argument')

        # add new kicktable to list or retrieve index of existing one with same fname.
        idx = _trackcpp.add_kicktable(filename)

        # update object attributes
        if idx != -1:
            self._status = _trackcpp.Status.success
            self._kicktable_idx = idx
            self._kicktable = _trackcpp.cvar.kicktable_list[idx]
        else:
            self._status = _trackcpp.Status.file_not_found
            self._kicktable_idx = -1
            self._kicktable = None

    @property
    def trackcpp_kickmap(self):
        """Return trackcpp Kickmap object."""
        return self._kicktable

    @property
    def kicktable_idx(self):
        """Return kicktable index in trackcpp kicktable_list vector."""
        return self._kicktable_idx

    @property
    def filename(self):
        """Filename corresponding to kicktable"""
        return self._kicktable.filename

    @property
    def length(self):
        """."""
        return self._kicktable.length

    @property
    def x_min(self):
        """."""
        return self._kicktable.x_min

    @property
    def x_max(self):
        """."""
        return self._kicktable.x_max

    @property
    def y_min(self):
        """."""
        return self._kicktable.y_min

    @property
    def y_max(self):
        """."""
        return self._kicktable.y_max

    @property
    def x_nrpts(self):
        """."""
        return self._kicktable.x_nrpts

    @property
    def y_nrpts(self):
        """."""
        return self._kicktable.y_nrpts

    @property
    def status(self):
        """Return last object status."""
        return self._status

    def get_kicks(self, rx, ry):
        """Return (hkick, vkick) at (rx,ry)."""
        idx = self.kicktable_idx
        self._status, hkick, vkick = _trackcpp.kicktable_getkicks_wrapper(idx, rx, ry)
        return hkick, vkick

    def __eq__(self, other):
        """."""
        if not isinstance(other, Kicktable):
            return NotImplemented
        for attr in self._kicktable.__swig_getmethods__:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


class Element(_Element):
    """."""

    def __init__(self, fam_name='', length=0.0):
        """."""
        super().__init__(fam_name, length)

    @property
    def kicktable(self):
        """."""
        if self.trackcpp_e.kicktable_idx != -1:
            kicktable = _trackcpp.cvar.kicktable_list[self.trackcpp_e.kicktable_idx]
            return Kicktable(kicktable=kicktable)
        else:
            return None

    @kicktable.setter
    def kicktable(self, value):
        """."""
        if not isinstance(value, Kicktable):
            raise TypeError('value must be of Kicktable type')
        self.trackcpp_e.kicktable = value._kicktable

    @property
    def rescale_kicks(self):
        """Scale factor applied to values given by kicktable.
        Default value: 1.0"""
        return self.trackcpp_e.rescale_kicks

    @rescale_kicks.setter
    def rescale_kicks(self, value):
        """."""
        self.trackcpp_e.rescale_kicks = value

    @property
    def KxL(self):
        """."""
        return -self.trackcpp_e.matrix66[1][0]

    @KxL.setter
    def KxL(self, value):
        """."""
        lst = list(self.trackcpp_e.matrix66[1])
        lst[0] = -value
        self.trackcpp_e.matrix66[1] = tuple(lst)

    @property
    def KyL(self):
        """."""
        return -self.trackcpp_e.matrix66[3][2]

    @KyL.setter
    def KyL(self, value):
        """."""
        lst = list(self.trackcpp_e.matrix66[3])
        lst[2] = -value
        self.trackcpp_e.matrix66[3] = tuple(lst)

    @property
    def Ks(self):
        """."""
        return -self.trackcpp_e.polynom_a[1]

    @Ks.setter
    def Ks(self, value):
        """."""
        self.trackcpp_e.polynom_a[1] = -value

    @property
    def KsL(self):
        """."""
        return -self.trackcpp_e.polynom_a[1] * self.trackcpp_e.length

    @KsL.setter
    def KsL(self, value):
        """."""
        self.trackcpp_e.polynom_a[1] = -value / self.trackcpp_e.length

    @property
    def KsxL(self):
        """."""
        return -self.trackcpp_e.matrix66[1][2]

    @KsxL.setter
    def KsxL(self, value):
        """."""
        lst = list(self.trackcpp_e.matrix66[1])
        lst[2] = -value
        self.trackcpp_e.matrix66[1] = tuple(lst)

    @property
    def KsyL(self):
        """."""
        return -self.trackcpp_e.matrix66[3][0]

    @KsyL.setter
    def KsyL(self, value):
        """."""
        lst = list(self.trackcpp_e.matrix66[3])
        lst[0] = -value
        self.trackcpp_e.matrix66[3] = tuple(lst)

    @property
    def matrix66(self):
        """."""
        return _numpy.array(self.trackcpp_e.matrix66)

    @matrix66.setter
    def matrix66(self, value):
        """."""
        tups = []
        for i in range(6):
            tups.append(tuple(float(value[i][j]) for j in range(6)))
        for i in range(6):
            self.trackcpp_e.matrix66[i] = tups[i]

    # --- private methods ---


_warnings.filterwarnings(
    "ignore", "Item size computed from the PEP 3118 \
    buffer format string does not match the actual item size.")
