"""Calculate Resonance Driving Terms."""

import numpy as _np

from .twiss import calc_twiss as _calc_twiss
from .. import lattice as _lattice


class FirstOrderDrivingTerms:
    """Calculate first order resonance driving terms."""

    GeometricTerms = "h10020", "h10110", "h10200", "h21000", "h30000"
    ChromaticTerms = "h11001", "h00111", "h20001", "h00201", "h10002"

    def __init__(self, accelerator, num_segments=2, energy_offset=0.0):
        """."""
        self._num_segments = num_segments
        self._accelerator = accelerator
        self._energy_offset = energy_offset
        self._driving_terms = dict()
        self._update_driving_terms()

    @property
    def accelerator(self):
        """."""
        return self._accelerator

    @accelerator.setter
    def accelerator(self, acc):
        """."""
        self._accelerator = acc
        self._update_driving_terms()

    @property
    def num_segments(self):
        """."""
        return self._num_segments

    @num_segments.setter
    def num_segments(self, num_segs):
        """."""
        self._num_segments = int(num_segs)
        self._update_driving_terms()

    @property
    def energy_offset(self):
        """."""
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, val):
        """."""
        self._energy_offset = float(val)
        self._update_driving_terms()

    @property
    def h11001(self):
        """."""
        return self._driving_terms["h11001"]

    @property
    def h00111(self):
        """."""
        return self._driving_terms["h00111"]

    @property
    def h20001(self):
        """."""
        return self._driving_terms["h20001"]

    @property
    def h00201(self):
        """."""
        return self._driving_terms["h00201"]

    @property
    def h10002(self):
        """."""
        return self._driving_terms["h10002"]

    @property
    def h21000(self):
        """."""
        return self._driving_terms["h21000"]

    @property
    def h30000(self):
        """."""
        return self._driving_terms["h30000"]

    @property
    def h10110(self):
        """."""
        return self._driving_terms["h10110"]

    @property
    def h10200(self):
        """."""
        return self._driving_terms["h10200"]

    @property
    def h10020(self):
        """."""
        return self._driving_terms["h10020"]

    def _update_driving_terms(self):
        quads = _np.array(_lattice.get_attribute(self._accelerator, 'KL'))
        sexts = _np.array(_lattice.get_attribute(self._accelerator, 'SL'))
        idcs = ~_np.isclose(quads, 0) | ~_np.isclose(sexts, 0)
        idcs = idcs.nonzero()[0]
        quads = quads[idcs] / self._num_segments
        sexts = sexts[idcs] / self._num_segments

        # Refine the lattice at the components that generate driving terms
        # The segmentation has to be in a way that the twiss functions are
        # calculated at the center of the slice. For instance, lets suppose
        # we will divide a given component in 3 equal contributions:
        #   :  first segment  : second segment  :  third segment  :
        #    ********|********:********|********:********|********
        #   ^        ^                 ^                 ^        ^
        # start twiss needed      twiss needed     twiss needed  end
        fractions = _np.r_[1, _np.ones(self._num_segments-1)*2, 1]
        fractions /= fractions.sum()
        acc = _lattice.refine_lattice(
            self._accelerator, indices=idcs, fractions=fractions
        )

        # get the indices
        _quads = _np.array(_lattice.get_attribute(acc, 'KL'))
        _sexts = _np.array(_lattice.get_attribute(acc, 'SL'))
        _idcs = ~_np.isclose(_quads, 0) | ~_np.isclose(_sexts, 0)
        _idcs = _idcs.nonzero()[0]
        _idcs = _idcs.reshape(idcs.size, -1)[:, 1:].ravel()

        quads = _np.tile(quads, (self._num_segments, 1)).T.ravel()
        sexts = _np.tile(sexts, (self._num_segments, 1)).T.ravel()
        twiss, *_ = _calc_twiss(
            acc, indices=_idcs, energy_offset=self._energy_offset
        )
        hx = _np.exp(1j*twiss.mux)
        hx2 = hx * hx
        hy2 = _np.exp(2j*twiss.muy)
        btx = twiss.betax
        bty = twiss.betay
        etx = twiss.etax

        setx = sexts * etx
        h11001 = +1/4 * (quads - 2 * setx) * btx
        h00111 = -1/4 * (quads - 2 * setx) * bty
        dr_terms = dict(
            h11001=h11001,
            h00111=h00111,
            h20001=h11001 * hx2 / 2,
            h00201=h00111 * hy2 / 2,
            h10002=1/2 * (quads - setx) * etx * _np.sqrt(btx) * hx,
        )

        sbx = -1/8 * sexts * _np.sqrt(btx) * hx
        sbxbx = sbx * btx
        sbxby = -1 * sbx * bty
        dr_terms.update(dict(
            h21000=sbxbx,
            h30000=sbxbx * hx2 / 3,
            h10110=2 * sbxby,
            h10200=sbxby * hy2,
            h10020=sbxby / hy2,
        ))

        for k, val in dr_terms.items():
            drt = _np.zeros(len(self._accelerator)+1, dtype=complex)
            drt[idcs] = val.reshape(idcs.size, -1).sum(axis=1)
            dr_terms[k] = _np.cumsum(drt)

        self._driving_terms = dr_terms
