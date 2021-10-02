"""Optics module."""

import numpy as _np

from .. import lattice as _lattice
from .. import tracking as _tracking
from ..utils import interactive as _interactive


@_interactive
def calc_edwards_teng(accelerator, energy_offset=0.0, indices='open'):
    """Perform linear analysis of coupled lattices.

    Notation is the same as in reference [3]

    References:
        [1] D.Edwars,L.Teng IEEE Trans.Nucl.Sci. NS-20, No.3, p.885-888, 1973
        [2] D.Sagan, D.Rubin Phys.Rev.Spec.Top.-Accelerators and beams,
            vol.2 (1999)
        [3] C.J. Gardner, Some Useful Linear Coupling Approximations.
            C-A/AP/#101 Brookhaven Nat. Lab. (July 2003)
        [4] Y-Luo. C-A/AP/#185 Brookhaven Nat. Lab. (january 2005)

    Args:
        accelerator (pyaccel.accelerator.Accelerator): lattice model
        energy_offset (float, optional): Energy Offset . Defaults to 0.0.
        indices : may be a ((list, tuple, numpy.ndarray), optional):
            list of element indices where closed orbit data is to be
            returned or a string:
                'open'  : return the closed orbit at the entrance of all
                    elements.
                'closed' : equal 'open' plus the orbit at the end of the last
                    element.
            If indices is None data will be returned only at the entrance
            of the first element. Defaults to 'open'.

    Returns:
        dict: with keys:
            spos (array, len(indices)x2x2) : longitudinal position [m]

            beta1 (array, len(indices)) : beta of first eigen-mode
            beta2 (array, len(indices)) : beta of second eigen-mode
            alpha1 (array, len(indices)) : alpha of first eigen-mode
            alpha2 (array, len(indices)) : alpha of second eigen-mode
            gamma1 (array, len(indices)) : gamma of first eigen-mode
            gamma2 (array, len(indices)) : gamma of second eigen-mode
            mu1 (array, len(indices)): phase advance of the first eigen-mode
            mu2 (array, len(indices)): phase advance of the second eigen-mode

            L1 (array, len(indices)x2x2) : matrices L1 in [3]
            L2 (array, len(indices)x2x2) : matrices L2 in [3]
            A (array, len(indices)x2x2) : matrices A in [3]
            B (array, len(indices)x2x2) : matrices B in [3]
            W (array, len(indices)x2x2) : matrices W in [3]
            d (array, len(indices)): d parameter in [3]

            min_tunesep (float) : estimative of minimum tune separation,
                Based on equation 85-87 of ref [3]:
                Assuming
                   mu1 = mux + minsep/2
                   mu2 = muy - minsep/2
                then, at mux = muy = mu0 ==> T = 0 ==> U = 2*sqrt(det(m+nbar))
                However, we know that U = 2*cos(mu1) - 2*cos(mu2)
                which yields, assuming mu0 ~ (mux + muy)/2
                sin(minsep/2) = sqrt(det(m+nbar))/sin(mu0)/2
            emit_ratio (float): estimative of invariant sharing ratio.
                Based on equation 258 of ref [3] we know that with weak
                coupling the invariant sharing is:
                    emit_ratio = (1 - d**2) / d**2

    """
    def Trans(a):
        return a.transpose(0, 2, 1)

    # 2-by-2 symplectic matrix
    S = _np.array([[0, 1], [-1, 0]])

    def SymTrans(a):
        if len(a.shape) >= 3:
            return -S @ Trans(a) @ S
        else:
            return -S @ a.T @ S

    spos = _lattice.find_spos(accelerator, indices=indices)
    m44, cum_mat = _tracking.find_m44(
        accelerator, energy_offset=energy_offset, indices=indices)

    # Calculate one turn matrix for every index in indices
    m44s = Trans(_np.linalg.solve(Trans(cum_mat), Trans(cum_mat @ m44)))
    M = m44s[:, :2, :2]
    N = m44s[:, 2:, 2:]
    m = m44s[:, 2:, :2]
    n = m44s[:, :2, 2:]

    t = _np.trace(M - N, axis1=1, axis2=2)

    m_plus_nbar = m + SymTrans(n)
    det_m_plus_nbar = _np.linalg.det(m_plus_nbar)

    u = _np.sign(t) * _np.sqrt(t*t + 4*det_m_plus_nbar)
    dsqr = (1 + t/u)/2
    d = _np.sqrt(dsqr)

    W_over_d = -m_plus_nbar/(dsqr*u)[:, None, None]

    W = -m_plus_nbar/(d*u)[:, None, None]
    A = M - n @ W_over_d
    B = N + W_over_d @ n

    mu1 = _np.arccos(_np.trace(A, axis1=1, axis2=2)/2)
    mu2 = _np.arccos(_np.trace(B, axis1=1, axis2=2)/2)
    beta1 = A[:, 0, 1]/_np.sin(mu1)
    beta2 = B[:, 0, 1]/_np.sin(mu2)
    gamma1 = -A[:, 1, 0]/_np.sin(mu1)
    gamma2 = -B[:, 1, 0]/_np.sin(mu2)
    alpha1 = (A[:, 0, 0] - _np.cos(mu1))/_np.sin(mu1)
    alpha2 = (B[:, 0, 0] - _np.cos(mu2))/_np.sin(mu2)

    # #### Determine Phase Advances #########

    # This method is based on equation 367 of ref[3]
    M11 = cum_mat[:, :2, :2]
    M12 = cum_mat[:, :2, 2:]
    M21 = cum_mat[:, 2:, :2]
    M22 = cum_mat[:, 2:, 2:]
    L1 = (d[0]*M11 - M12@W[0]) / d[:, None, None]
    L2 = (d[0]*M22 + M21@SymTrans(W[0])) / d[:, None, None]

    # To get the phase advances we use the same logic employed in
    # method calc_twiss of trackcpp:
    mu1 = _np.arctan(L1[:, 0, 1]/(
        L1[:, 0, 0]*beta1[0] - L1[:, 0, 1]*alpha1[0]))
    mu2 = _np.arctan(L2[:, 0, 1]/(
        L2[:, 0, 0]*beta2[0] - L2[:, 0, 1]*alpha2[0]))
    # unwrap phases
    summ = _np.zeros(mu1.size)
    _np.cumsum(_np.diff(mu1) < 0, out=summ[1:])
    mu1 += summ * _np.pi
    _np.cumsum(_np.diff(mu2) < 0, out=summ[1:])
    mu2 += summ * _np.pi

    # ###### Estimative of emittance ratio #######
    emit_ratio = (1-dsqr)/dsqr

    # ###### Estimate Minimum tune separation #####
    mux = _np.arccos(_np.trace(M[0])/2)
    muy = _np.arccos(_np.trace(N[0])/2)
    mu0 = (mux + muy)/2
    min_tunesep = 2*_np.arcsin(
        _np.sqrt(_np.abs(det_m_plus_nbar))/_np.sin(mu0)/2)
    min_tunesep /= 2*_np.pi

    return {
        'spos': spos,
        'beta1': beta1, 'beta2': beta2,
        'alpha1': alpha1, 'alpha2': alpha2,
        'gamma1': gamma1, 'gamma2': gamma2,
        'mu1': mu1, 'mu2': mu2,
        'L1': L1, 'L2': L2,
        'd': d, 'A': A, 'B': B, 'W': W,
        'min_tunesep': min_tunesep[0],
        'emit_ratio': emit_ratio[0],
        }
