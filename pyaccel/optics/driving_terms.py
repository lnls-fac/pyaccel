"""Calculate Resonance Driving Terms."""

import matplotlib.pyplot as _mplt
import numpy as _np

from .. import lattice as _lattice
from ..graphics import draw_lattice as _draw_lattice
from .twiss import calc_twiss as _calc_twiss


class FirstOrderDrivingTerms:
    """Calculate first order resonance driving terms.

    The implementation is based on Bengtsson report:
        https://ados.web.psi.ch/slsnotes/sls0997.pdf
    """

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

    def plot_rdt_along_ring(self, geometric=True, symmetry=1):
        """Plot RDTs along ring.

        Args:
            geometric (bool, optional): Whether to plot geometric or chromatic
                RDTs. Defaults to True.
            symmetry (int, optional): Symmetry of the ring. used to define
                range of the plot. Defaults to 1.

        Returns:
            fig: figure object.
            axs: axes of the figure.
        """
        fig, (ay, ax) = _mplt.subplots(
            2,
            1,
            height_ratios=[1, 10],
            figsize=(8, 4),
            sharex=True,
            gridspec_kw=dict(
                hspace=0.01, right=0.99, left=0.1, top=0.95,
            )
        )
        mod = self._accelerator
        pos = _lattice.find_spos(mod, indices='closed')

        terms = self.GeometricTerms if geometric else self.ChromaticTerms
        for term in terms:
            ax.plot(pos, _np.abs(getattr(self, term)), label=term)

        _draw_lattice(mod, gca=ay)

        title = 'Geometric' if geometric else 'Chromatic'
        unit = '[m$^{-1/2}$]' if geometric else '[1 or m$^{1/2}$]'
        ay.set_title(title + ' Driving Terms')
        ax.set_ylabel(r'Abs $\left(h_{abcd0}\right)$ ' + unit)
        ax.set_xlabel('Position [m]')
        ax.grid(True, alpha=0.4, ls='--', lw=1)
        ay.set_ylim(-1, 1)
        ay.set_xlim(0, mod.length/symmetry)
        ay.set_axis_off()
        ax.legend(loc='best', ncol=3)

        return fig, (ay, ax)

    def plot_one_turn_rdts(self, geometric=True):
        """Plot RDTs after one turn.

        Args:
            geometric (bool, optional): Whether to plot geometric or chromatic
                RDTs. Defaults to True.

        Returns:
            fig: figure object.
            ax: axis of the figure.
        """
        fig, ax = _mplt.subplots(1, 1, figsize=(4, 3))

        colors = _mplt.rcParams["axes.prop_cycle"].by_key()["color"]
        terms = self.GeometricTerms if geometric else self.ChromaticTerms
        for cor, drt_l in zip(colors, terms):
            drt = getattr(self, drt_l)[-1]
            ax.plot(drt.real, drt.imag, 'o', color=cor, label=drt_l)
            ax.annotate(
                "",
                xy=(drt.real, drt.imag),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>", color=cor, mutation_scale=10
                )
            )
        title = 'Geometric' if geometric else 'Chromatic'
        unit = '[m$^{-1/2}$]' if geometric else '[1 or m$^{1/2}$]'
        ax.set_title(title + ' One Turn RDTs')
        ax.grid(True, alpha=0.3, ls='--', lw=1)
        ax.legend(loc='best', fontsize='small')
        ax.set_ylabel('Imaginary ' + unit)
        ax.set_xlabel('Real ' + unit)
        fig.tight_layout()
        return fig, ax

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

    @staticmethod
    def calc_rdt_types_for_potential(
        potential='sext_norm',
        include_complex_conjugates=True,
        consider_closed_orbit=True,
        consider_vertical_dispersion=True
    ):
        """Calculate RDTs for a given potential.

        Args:
            potential (str, optional): type of potential to consider. Defaults
                to 'sext_norm'. Possible options are: 'sext_norm', 'quad_norm'
                and 'quad_skew'.
            include_complex_conjugates (bool, optional): Whether or not to
                return the RDTS that are complex conjugates of others.
                Defaults to True.
            consider_closed_orbit (bool, optional): Whether or not to consider
                closed orbit errors on RDTs calculations. Closed orbit errors
                may lead to ressonances via multipole feeddown.
                Defaults to True.
            consider_vertical_dispersion (bool, optional): Whether to consider
                the vertical dispersion on calculation of RDTs.
                Defaults to True.

        Returns:
            rdts: list of tuples indicating the powers of h+, h-, delta and so
                on.
            rdt_coefs: list of floats with the multiplicative factor of a
                given RDT.
        """
        import sympy as smp

        # Define the variables
        sl, xc, yc = smp.symbols('SL x_c y_c', real=True)
        x, y, delta = smp.symbols('x y delta', real=True)
        etax, etay, betax, betay = smp.symbols(
            'eta_x eta_y beta_x beta_y', real=True
        )
        mux, muy = smp.symbols('mu_x mu_y', real=True)
        hxp, hxn, hyp, hyn = smp.symbols('h_x^+ h_x^- h_y^+ h_y^-', real=False)

        # Define the potential and apply affine map to coordinates:
        pots = {
            'quad_skew': -sl / 2 * (x * x - y * y) * (1 - delta),
            'quad_norm': -sl * x * y * (1 - delta),
            'sext_norm': -sl / 3 * x * (x * x - 3 * y * y),
        }
        pot = pots[potential]

        _xc = consider_closed_orbit * xc
        _yc = consider_closed_orbit * yc
        _ety = consider_vertical_dispersion * etay
        pot = pot.subs(x, smp.sqrt(betax) * x + _xc + etax * delta)
        pot = pot.subs(y, smp.sqrt(betay) * y + _yc + _ety * delta)
        pot = pot.subs(x, (hxp + hxn)/2)
        pot = pot.subs(y, (hyp + hyn)/2)
        pot = pot.subs(hxp, hxp * smp.exp(1j*mux))
        pot = pot.subs(hxn, hxn * smp.exp(-1j*mux))
        pot = pot.subs(hyp, hyp * smp.exp(1j*muy))
        pot = pot.subs(hyn, hyn * smp.exp(-1j*muy))

        # Gather the driving terms:
        var_lst = [hxp, hxn, hyp, hyn, delta, etax, etay, ]
        rdts = []
        rdt_coefs = []
        for trm in pot.expand().as_ordered_terms():
            rdt = []
            rdt_coef, coef_lst = trm.as_coeff_mul()
            rdt_coefs.append(rdt_coef)
            for v in var_lst:
                for coef in coef_lst:
                    coef, powe = coef.as_base_exp()
                    if coef == v:
                        rdt.append(powe)
                        break
                else:
                    rdt.append(0)
            rdts.append(rdt)

        # Select only complex conjugates:
        if include_complex_conjugates:
            return rdts, rdt_coefs

        # Find which driving terms are complex conjugates of others
        is_cc = [False, ] * len(rdts)
        for i1, rdt in enumerate(rdts):
            for i2, rdt2 in enumerate(rdts[i1+1:], i1+1):
                xp, xm, yp, ym, dt, ex, ey = rdt2
                rdt2 = [xm, xp, ym, yp, dt, ex, ey]
                if (xm != xp or ym != yp) and rdt2 == rdt:
                    is_cc[i2] = True

        rdts = [rdt for idx, rdt in zip(is_cc, rdts) if not idx]
        rdt_coefs = [rdt for idx, rdt in zip(is_cc, rdts) if not idx]
        return rdts, rdt_coefs
