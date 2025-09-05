"""This module to performs linear analysis of coupled lattices.

Notation is the same as in reference [3]
In case some strange results appear in phase advances or beta functions,
the reading of [2] is encouraged, since the authors discuss subtleties not
addressed here for strong coupled motion.

References:
    [1] Edwards, D. A., & Teng, L. C. (1973). Parametrization of Linear
        Coupled Motion in Periodic Systems. IEEE Transactions on Nuclear
        Science, 20(3), 885–888. https://doi.org/10.1109/TNS.1973.4327279
    [2] Sagan, D., & Rubin, D. (1999). Linear analysis of coupled
        lattices. Physical Review Special Topics - Accelerators and Beams,
        2(7), 22–26. https://doi.org/10.1103/physrevstab.2.074001
    [3] C.J. Gardner, Some Useful Linear Coupling Approximations.
        C-A/AP/#101 Brookhaven Nat. Lab. (July 2003)

"""

import numpy as _np
from mathphys.functions import get_namedtuple as _get_namedtuple

from .. import lattice as _lattice, tracking as _tracking
from ..utils import interactive as _interactive
from .miscellaneous import OpticsError as _OpticsError


class EdwardsTeng(_np.record):
    """Edwards and Teng decomposition of the transfer matrices.

    Notation is the same as in reference [3].
    In case some strange results appear in phase advances or beta functions,
    the reading of [2] is encouraged, since the authors discuss subtleties not
    addressed here for strong coupled motion.

    References:
        [1] Edwards, D. A., & Teng, L. C. (1973). Parametrization of Linear
            Coupled Motion in Periodic Systems. IEEE Transactions on Nuclear
            Science, 20(3), 885–888. https://doi.org/10.1109/TNS.1973.4327279
        [2] Sagan, D., & Rubin, D. (1999). Linear analysis of coupled
            lattices. Physical Review Special Topics - Accelerators and Beams,
            2(7), 22–26. https://doi.org/10.1103/physrevstab.2.074001
        [3] C.J. Gardner, Some Useful Linear Coupling Approximations.
            C-A/AP/#101 Brookhaven Nat. Lab. (July 2003)

    Contains the decomposed parameters:
        spos (array, len(indices)x2x2) : longitudinal position [m]
        beta1 (array, len(indices)) : beta of first eigen-mode
        beta2 (array, len(indices)) : beta of second eigen-mode
        alpha1 (array, len(indices)) : alpha of first eigen-mode
        alpha2 (array, len(indices)) : alpha of second eigen-mode
        gamma1 (array, len(indices)) : gamma of first eigen-mode
        gamma2 (array, len(indices)) : gamma of second eigen-mode
        mu1 (array, len(indices)): phase advance of the first eigen-mode
        mu2 (array, len(indices)): phase advance of the second eigen-mode
        W (array, len(indices)x2x2) : matrices W in [3]

    """

    DTYPE = '<f8'
    ORDER = _get_namedtuple(
        'Order',
        field_names=[
            'spos',
            'beta1',
            'alpha1',
            'mu1',
            'beta2',
            'alpha2',
            'mu2',
            'W_11',
            'W_12',
            'W_21',
            'W_22',
            'eta1',
            'etap1',
            'eta2',
            'etap2',
            'rx',
            'px',
            'ry',
            'py',
            'de',
            'dl',
        ],
    )

    def __setattr__(self, attr, val):
        """."""
        if attr == 'co':
            self._set_co(val)
        else:
            super().__setattr__(attr, val)

    def __str__(self):
        """."""
        rst = ''
        rst += 'spos          : ' + '{0:+10.3e}'.format(self.spos)
        fmt = '{0:+10.3e}, {1:+10.3e}'
        rst += '\nrx, ry        : ' + fmt.format(self.rx, self.ry)
        rst += '\npx, py        : ' + fmt.format(self.px, self.py)
        rst += '\nde, dl        : ' + fmt.format(self.de, self.dl)
        rst += '\nmu1, mu2      : ' + fmt.format(self.mu1, self.mu2)
        rst += '\nbeta1, beta2  : ' + fmt.format(self.beta1, self.beta2)
        rst += '\nalpha1, alpha2: ' + fmt.format(self.alpha1, self.alpha2)
        rst += '\neta1, eta2    : ' + fmt.format(self.eta1, self.eta2)
        rst += '\netap1, etap2  : ' + fmt.format(self.etap1, self.etap2)
        return rst

    @property
    def co(self):
        """Closed-Orbit in XY plane coordinates.

        Returns:
            numpy.ndarray (6, ): 6D phase space point around matrices were
                calculated.

        """
        return _np.array(
            [self.rx, self.px, self.ry, self.py, self.de, self.dl]
        )

    @property
    def W(self):
        """2D mixing matrix from ref [3].

        Returns:
            numpy.ndarray (2x2): W matrix from ref [3].

        """
        return _np.array([[self.W_11, self.W_12], [self.W_21, self.W_22]])

    @W.setter
    def W(self, val):
        self[EdwardsTeng.ORDER.W_11] = val[0, 0]
        self[EdwardsTeng.ORDER.W_12] = val[0, 1]
        self[EdwardsTeng.ORDER.W_21] = val[1, 0]
        self[EdwardsTeng.ORDER.W_22] = val[1, 1]

    @property
    def d(self):
        """Parameter d from ref [3], calculated via equation 81.

        Returns:
            float: d from ref [3].

        """
        return _np.sqrt(1 - _np.linalg.det(self.W))

    @property
    def R(self):
        """4D matrix that transforms from normal modes to XY plane.

        Returns:
            numpy.ndarray (4x4): R matrix from ref [3].

        """
        deyes = self.d * _np.eye(2)
        return _np.block(
            [[deyes, _symplectic_transpose(self.W)], [-self.W, deyes]]
        )

    @property
    def Rinv(self):
        """4D matrix that transforms from XY plane to normal modes.

        Returns:
            numpy.ndarray (4x4): Rinv matrix from ref [3].

        """
        deyes = self.d * _np.eye(2)
        return _np.block(
            [[deyes, -_symplectic_transpose(self.W)], [self.W, deyes]]
        )

    def from_normal_modes(self, pos):
        """Transform from normal modes to XY plane.

        Args:
            pos (numpy.ndarray): (4, N) or (6, N) positions in phase space in
                normal modes coordinates.

        Returns:
            pos (numpy.ndarray): (4, N) or (6, N) positions in phase space in
                XY coordinates.

        """
        pos = pos.copy()
        pos[:4] = self.R @ pos[:4]
        return pos

    def to_normal_modes(self, pos):
        """Transform from XY plane to normal modes.

        Args:
            pos (numpy.ndarray): (4, N) or (6, N) positions in phase space in
                XY coordinates.

        Returns:
            pos (numpy.ndarray): (4, N) or (6, N) positions in phase space in
                normal mode coordinates.

        """
        pos = pos.copy()
        pos[:4] = self.Rinv @ pos[:4]
        return pos

    def make_dict(self):
        """."""
        cod = self.co
        beta = [self.beta1, self.beta2]
        alpha = [self.alpha1, self.alpha2]
        eta = [self.eta1, self.eta2]
        etap = [self.etap1, self.etap2]
        mus = [self.mu1, self.mu2]
        return {
            'co': cod,
            'beta': beta,
            'alpha': alpha,
            'eta': eta,
            'etap': etap,
            'mu': mus,
        }

    @staticmethod
    def make_new(*args, **kwrgs):
        """Build a Twiss object."""
        if args:
            if isinstance(args[0], dict):
                kwrgs = args[0]
        twi = EdwardsTengArray(1)
        cod = kwrgs.get('co', (0.0,) * 6)
        twi['rx'], twi['px'], twi['ry'], twi['py'], twi['de'], twi['dl'] = cod
        twi['mu1'], twi['mu2'] = kwrgs.get('mu', (0.0, 0.0))
        twi['beta1'], twi['beta2'] = kwrgs.get('beta', (0.0, 0.0))
        twi['alpha1'], twi['alpha2'] = kwrgs.get('alpha', (0.0, 0.0))
        twi['eta1'], twi['eta2'] = kwrgs.get('eta', (0.0, 0.0))
        twi['etap1'], twi['etap2'] = kwrgs.get('etap', (0.0, 0.0))
        return twi[0]

    def _set_co(self, value):
        """."""
        try:
            leng = len(value)
        except TypeError:
            leng = 6
            value = [value] * leng
        if leng != 6:
            raise ValueError('closed orbit must have 6 elements.')
        self[EdwardsTeng.ORDER.rx] = value[0]
        self[EdwardsTeng.ORDER.px] = value[1]
        self[EdwardsTeng.ORDER.ry] = value[2]
        self[EdwardsTeng.ORDER.py] = value[3]
        self[EdwardsTeng.ORDER.de] = value[4]
        self[EdwardsTeng.ORDER.dl] = value[5]


class EdwardsTengArray(_np.ndarray):
    """Array of Edwards and Teng objects.

    Notation is the same as in reference [3]
    In case some strange results appear in phase advances or beta functions,
    the reading of [2] is encouraged, since the authors discuss subtleties not
    addressed here for strong coupled motion.

    References:
        [1] Edwards, D. A., & Teng, L. C. (1973). Parametrization of Linear
            Coupled Motion in Periodic Systems. IEEE Transactions on Nuclear
            Science, 20(3), 885–888. https://doi.org/10.1109/TNS.1973.4327279
        [2] Sagan, D., & Rubin, D. (1999). Linear analysis of coupled
            lattices. Physical Review Special Topics - Accelerators and Beams,
            2(7), 22–26. https://doi.org/10.1103/physrevstab.2.074001
        [3] C.J. Gardner, Some Useful Linear Coupling Approximations.
            C-A/AP/#101 Brookhaven Nat. Lab. (July 2003)

    Contains the decomposed parameters:
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
        W (array, len(indices)x2x2) : matrices W in [3]
        d (array, len(indices)): d parameter in [3]

    """

    def __eq__(self, other):
        """."""
        return _np.all(super().__eq__(other))

    def __new__(cls, edteng=None, copy=True):
        """."""
        length = 1
        if isinstance(edteng, (int, _np.int_)):
            length = edteng
            edteng = None
        elif isinstance(edteng, EdwardsTengArray):
            return edteng.copy() if copy else edteng

        if edteng is None:
            arr = _np.zeros(
                (length, len(EdwardsTeng.ORDER)), dtype=EdwardsTeng.DTYPE
            )
        elif isinstance(edteng, _np.ndarray):
            arr = edteng.copy() if copy else edteng
        elif isinstance(edteng, _np.record):
            arr = _np.ndarray(
                (edteng.size, len(EdwardsTeng.ORDER)), buffer=edteng.data
            )
            arr = arr.copy() if copy else arr

        fmts = [(fmt, EdwardsTeng.DTYPE) for fmt in EdwardsTeng.ORDER._fields]
        return super().__new__(
            cls, shape=(arr.shape[0],), dtype=(EdwardsTeng, fmts), buffer=arr
        )

    @property
    def spos(self):
        """."""
        return self['spos']

    @spos.setter
    def spos(self, value):
        self['spos'] = value

    @property
    def beta1(self):
        """."""
        return self['beta1']

    @beta1.setter
    def beta1(self, value):
        self['beta1'] = value

    @property
    def alpha1(self):
        """."""
        return self['alpha1']

    @alpha1.setter
    def alpha1(self, value):
        self['alpha1'] = value

    @property
    def gamma1(self):
        """."""
        return (1 + self['alpha1'] * self['alpha1']) / self['beta1']

    @property
    def mu1(self):
        """."""
        return self['mu1']

    @mu1.setter
    def mu1(self, value):
        self['mu1'] = value

    @property
    def beta2(self):
        """."""
        return self['beta2']

    @beta2.setter
    def beta2(self, value):
        self['beta2'] = value

    @property
    def alpha2(self):
        """."""
        return self['alpha2']

    @alpha2.setter
    def alpha2(self, value):
        self['alpha2'] = value

    @property
    def gamma2(self):
        """."""
        return (1 + self['alpha2'] * self['alpha2']) / self['beta2']

    @property
    def mu2(self):
        """."""
        return self['mu2']

    @mu2.setter
    def mu2(self, value):
        self['mu2'] = value

    @property
    def W_11(self):
        """."""
        return self['W_11']

    @W_11.setter
    def W_11(self, val):
        self['W_11'] = val

    @property
    def W_12(self):
        """."""
        return self['W_12']

    @W_12.setter
    def W_12(self, val):
        self['W_12'] = val

    @property
    def W_21(self):
        """."""
        return self['W_21']

    @W_21.setter
    def W_21(self, val):
        self['W_21'] = val

    @property
    def W_22(self):
        """."""
        return self['W_22']

    @W_22.setter
    def W_22(self, val):
        self['W_22'] = val

    @property
    def W(self):
        """2D mixing matrix from ref [3].

        Returns:
            numpy.ndarray (Nx2x2): W matrix from ref [3].

        """
        mat = _np.zeros((self.W_11.size, 2, 2))
        mat[:, 0, 0] = self.W_11
        mat[:, 0, 1] = self.W_12
        mat[:, 1, 0] = self.W_21
        mat[:, 1, 1] = self.W_22
        return mat

    @W.setter
    def W(self, value):
        self.W_11 = value[:, 0, 0]
        self.W_12 = value[:, 0, 1]
        self.W_21 = value[:, 1, 0]
        self.W_22 = value[:, 1, 1]

    @property
    def d(self):
        """Parameter d from ref [3], calculated via equation 81.

        Returns:
            numpy.ndarray (N, ): d from ref [3].

        """
        return _np.sqrt(1 - _np.linalg.det(self.W))

    @property
    def R(self):
        """4D matrix that transforms from normal modes to XY plane.

        Returns:
            numpy.ndarray (Nx4x4): R matrix from ref [3].

        """
        deyes = self.d[:, None, None] * _np.eye(2)
        return _np.block(
            [[deyes, _symplectic_transpose(self.W)], [-self.W, deyes]]
        )

    @property
    def Rinv(self):
        """4D matrix that transforms from XY plane to normal modes.

        Returns:
            numpy.ndarray (Nx4x4): Rinv matrix from ref [3].

        """
        deyes = self.d[:, None, None] * _np.eye(2)
        return _np.block(
            [[deyes, -_symplectic_transpose(self.W)], [self.W, deyes]]
        )

    def from_normal_modes(self, pos):
        """Transform from normal modes to XY plane.

        Args:
            pos (numpy.ndarray): (4, len(self)) or (6, len(self)) positions in
                phase space in normal modes coordinates.

        Returns:
            numpy.ndarray: (4, len(self)) or (6, len(self)) positions in phase
                space in XY coordinates.

        """
        pos = pos.copy()
        pos[:4] = _np.einsum('ijk,ki->ji', self.R, pos[:4])
        return pos

    def to_normal_modes(self, pos):
        """Transform from XY plane to normal modes.

        Args:
            pos (numpy.ndarray): (4, len(self)) or (6, len(self)) positions in
                phase space in XY coordinates.

        Returns:
            numpy.ndarray: (4, len(self)) or (6, len(self)) positions in phase
                space in normal mode coordinates.

        """
        pos = pos.copy()
        pos[:4] = _np.einsum('ijk,ki->ji', self.Rinv, pos[:4])
        return pos

    @property
    def eta1(self):
        """."""
        return self['eta1']

    @eta1.setter
    def eta1(self, value):
        self['eta1'] = value

    @property
    def etap1(self):
        """."""
        return self['etap1']

    @etap1.setter
    def etap1(self, value):
        self['etap1'] = value

    @property
    def eta2(self):
        """."""
        return self['eta2']

    @eta2.setter
    def eta2(self, value):
        self['eta2'] = value

    @property
    def etap2(self):
        """."""
        return self['etap2']

    @etap2.setter
    def etap2(self, value):
        self['etap2'] = value

    @property
    def rx(self):
        """."""
        return self['rx']

    @rx.setter
    def rx(self, value):
        self['rx'] = value

    @property
    def px(self):
        """."""
        return self['px']

    @px.setter
    def px(self, value):
        self['px'] = value

    @property
    def ry(self):
        """."""
        return self['ry']

    @ry.setter
    def ry(self, value):
        self['ry'] = value

    @property
    def py(self):
        """."""
        return self['py']

    @py.setter
    def py(self, value):
        self['py'] = value

    @property
    def de(self):
        """."""
        return self['de']

    @de.setter
    def de(self, value):
        self['de'] = value

    @property
    def dl(self):
        """."""
        return self['dl']

    @dl.setter
    def dl(self, value):
        self['dl'] = value

    @property
    def co(self):
        """Trajectory in XY plane coordinates.

        Returns:
            numpy.ndarray (6, ): 6D phase space trajectory around matrices were
                calculated.

        """
        return _np.array(
            [self.rx, self.px, self.ry, self.py, self.de, self.dl]
        )

    @co.setter
    def co(self, value):
        """."""
        self.rx, self.px = value[0], value[1]
        self.ry, self.py = value[2], value[3]
        self.de, self.dl = value[4], value[5]

    @staticmethod
    def compose(edteng_list):
        """."""
        if isinstance(edteng_list, (list, tuple)):
            for val in edteng_list:
                if not isinstance(val, (EdwardsTeng, EdwardsTengArray)):
                    raise _OpticsError(
                        'can only compose lists of Twiss objects.'
                    )
        else:
            raise _OpticsError('can only compose lists of Twiss objects.')

        arrs = list()
        for val in edteng_list:
            arrs.append(
                _np.ndarray(
                    (val.size, len(EdwardsTeng.ORDER)), buffer=val.data
                )
            )
        arrs = _np.vstack(arrs)
        return EdwardsTengArray(arrs)


@_interactive
def calc_edwards_teng(
    accelerator=None, init_edteng=None, indices='open', energy_offset=0.0
):
    """Perform linear analysis of coupled lattices.

    Notation is the same as in reference [3]
    In case some strange results appear in phase advances or beta functions,
    the reading of [2] is encouraged, since the authors discuss subtleties not
    addressed here for strong coupled motion.

    References:
        [1] Edwards, D. A., & Teng, L. C. (1973). Parametrization of Linear
            Coupled Motion in Periodic Systems. IEEE Transactions on Nuclear
            Science, 20(3), 885–888. https://doi.org/10.1109/TNS.1973.4327279
        [2] Sagan, D., & Rubin, D. (1999). Linear analysis of coupled
            lattices. Physical Review Special Topics - Accelerators and Beams,
            2(7), 22–26. https://doi.org/10.1103/physrevstab.2.074001
        [3] C.J. Gardner, Some Useful Linear Coupling Approximations.
            C-A/AP/#101 Brookhaven Nat. Lab. (July 2003)

    Args:
        accelerator (pyaccel.accelerator.Accelerator): lattice model
        init_edteng (pyaccel.optics.EdwardsTeng, optional): EdwardsTeng
            parameters at the start of first element. Defaults to None.
        indices : may be a ((list, tuple, numpy.ndarray), optional):
            list of element indices where closed orbit data is to be
            returned or a string:
                'open'  : return the closed orbit at the entrance of all
                    elements.
                'closed' : equal 'open' plus the orbit at the end of the last
                    element.
            If indices is None data will be returned only at the entrance
            of the first element. Defaults to 'open'.
        energy_offset (float, optional): float denoting the energy deviation
            (used only for periodic solutions). Defaults to 0.0.

    Returns:
        pyaccel.optics.EdwardsTengArray : array of decompositions of the
            transfer matrices
        numpy.ndarray (4x4): transfer matrix of the line/ring

    """
    acc = accelerator
    if acc.cavity_on or acc.radiation_on:
        raise RuntimeError(
            'Edwards and Teng decomposition is restricted to symplectic 4D '
            'motion. Please turn off `cavity_on` and `radiation_on` flags.'
        )

    cod = None
    if init_edteng is not None:
        if energy_offset:
            raise _OpticsError(
                'energy_offset and init_teng are mutually exclusive options. '
                'Add the desired energy deviation to the appropriate '
                'position  of init_edteng object.'
            )
        # as a transport line: uses init_edteng
        fixed_point = init_edteng.co
    else:
        # as a periodic system: try to find periodic solution
        if acc.harmonic_number == 0:
            raise _OpticsError(
                'Either harmonic number was not set or calc_edwards_teng was '
                'invoked for transport line without initial  EdwardsTeng'
            )
        cod = _tracking.find_orbit(
            acc, energy_offset=energy_offset, indices='closed'
        )
        fixed_point = cod[:, 0]

    m44, cum_mat = _tracking.find_m44(
        acc, indices='closed', fixed_point=fixed_point
    )

    indices = _tracking._process_indices(acc, indices)
    edteng = EdwardsTengArray(cum_mat.shape[0])
    edteng.spos = _lattice.find_spos(acc, indices='closed')

    # Calculate initial matrices:
    #  Based on ref [3].
    if init_edteng is None:
        # Eqs 7 and 8
        M = m44[:2, :2]
        N = m44[2:4, 2:4]
        m = m44[2:4, :2]
        n = m44[:2, 2:4]

        # Eq 86
        t = _np.trace(M - N)

        # Eq 87
        m_plus_nbar = m + _symplectic_transpose(n)
        det_m_plus_nbar = _np.linalg.det(m_plus_nbar)
        u = _np.sign(t) * _np.sqrt(t * t + 4 * det_m_plus_nbar)

        # Eq 86
        dsqr = (1 + t / u) / 2

        # Eq 85
        W_over_d = -m_plus_nbar / (dsqr * u)
        d0 = _np.sqrt(dsqr)
        W0 = -m_plus_nbar / (d0 * u)

        # Eq 85 with notation of Eq XX
        A0 = M - n @ W_over_d
        B0 = N + W_over_d @ n

        # Get initial betas and alphas. (Based on calc_twiss of trackcpp.)
        sin_mu1 = _np.sign(A0[0, 1]) * _np.sqrt(
            -A0[0, 1] * A0[1, 0] - (A0[0, 0] - A0[1, 1]) ** 2 / 4.0
        )
        sin_mu2 = _np.sign(B0[0, 1]) * _np.sqrt(
            -B0[0, 1] * B0[1, 0] - (B0[0, 0] - B0[1, 1]) ** 2 / 4.0
        )
        alpha10 = (A0[0, 0] - A0[1, 1]) / 2 / sin_mu1
        alpha20 = (B0[0, 0] - B0[1, 1]) / 2 / sin_mu2
        beta10 = A0[0, 1] / sin_mu1
        beta20 = B0[0, 1] / sin_mu2
    else:
        W0 = init_edteng.W
        d0 = init_edteng.d
        alpha10 = init_edteng.alpha1
        alpha20 = init_edteng.alpha2
        beta10 = init_edteng.beta1
        beta20 = init_edteng.beta2

    # #### Determine Twiss Parameters of uncoupled motion #########
    # First get the initial transfer matrices decompositions.
    # Eq 362
    M11 = cum_mat[:, :2, :2]
    M12 = cum_mat[:, :2, 2:4]
    M21 = cum_mat[:, 2:4, :2]
    M22 = cum_mat[:, 2:4, 2:4]

    # Eq 366
    L1_unnorm = d0 * M11 - M12 @ W0
    d = _np.sqrt(_np.linalg.det(L1_unnorm))
    # Eq 367
    L1 = L1_unnorm / d[:, None, None]
    L2 = (d0 * M22 + M21 @ _symplectic_transpose(W0)) / d[:, None, None]
    # Eq 368
    edteng.W = -(d0 * M21 - M22 @ W0) @ _symplectic_transpose(L1)

    # Eq 369: we don't need to calculate this.
    # A = L1 @ A0 @ _symplectic_transpose(L1)
    # B = L2 @ B0 @ _symplectic_transpose(L2)

    # Get optical functions along the ring (Based on calc_twiss of trackcpp):
    edteng.beta1 = (
        (L1[:, 0, 0] * beta10 - L1[:, 0, 1] * alpha10) ** 2 + L1[:, 0, 1] ** 2
    ) / beta10
    edteng.beta2 = (
        (L2[:, 0, 0] * beta20 - L2[:, 0, 1] * alpha20) ** 2 + L2[:, 0, 1] ** 2
    ) / beta20
    edteng.alpha1 = (
        -(
            (L1[:, 0, 0] * beta10 - L1[:, 0, 1] * alpha10)
            * (L1[:, 1, 0] * beta10 - L1[:, 1, 1] * alpha10)
            + L1[:, 0, 1] * L1[:, 1, 1]
        )
        / beta10
    )
    edteng.alpha2 = (
        -(
            (L2[:, 0, 0] * beta20 - L2[:, 0, 1] * alpha20)
            * (L2[:, 1, 0] * beta20 - L2[:, 1, 1] * alpha20)
            + L2[:, 0, 1] * L2[:, 1, 1]
        )
        / beta20
    )
    mu1 = _np.arctan(
        L1[:, 0, 1] / (L1[:, 0, 0] * beta10 - L1[:, 0, 1] * alpha10)
    )
    mu2 = _np.arctan(
        L2[:, 0, 1] / (L2[:, 0, 0] * beta20 - L2[:, 0, 1] * alpha20)
    )
    # unwrap phases
    summ = _np.zeros(mu1.size)
    _np.cumsum(_np.diff(mu1) < 0, out=summ[1:])
    mu1 += summ * _np.pi
    _np.cumsum(_np.diff(mu2) < 0, out=summ[1:])
    mu2 += summ * _np.pi
    edteng.mu1 = mu1
    edteng.mu2 = mu2

    # ####### Handle dispersion function and orbit #######:
    dp = 1e-6
    if cod is not None:
        coddp = _tracking.find_orbit(
            acc, energy_offset=fixed_point[4] + dp, indices='closed'
        )
    else:
        cod, *_ = _tracking.line_pass(acc, fixed_point, indices='closed')
        etas_norm = _np.array(
            [
                init_edteng.eta1,
                init_edteng.etap1,
                init_edteng.eta2,
                init_edteng.etap2,
            ]
        )
        etas = init_edteng.from_normal_modes(etas_norm)
        fixed_point[:4] += dp * etas
        fixed_point[4] += dp
        coddp, *_ = _tracking.line_pass(acc, fixed_point, indices='closed')

    eta = (coddp - cod) / dp
    eta = edteng.to_normal_modes(eta)
    edteng.co = cod
    edteng.eta1 = eta[0]
    edteng.etap1 = eta[1]
    edteng.eta2 = eta[2]
    edteng.etap2 = eta[3]

    return edteng[indices], m44


def estimate_coupling_parameters(edteng_end):
    """Estimate minimum tune separation and emittance ratio.

    The estimative uses Edwards and Teng decomposition of the one turn matrix.

    Notation is the same as in reference [3]
    Reading of [2] is encouraged, since the authors discuss subtleties not
    addressed here for strong coupled motion.

    References:
        [1] Edwards, D. A., & Teng, L. C. (1973). Parametrization of Linear
            Coupled Motion in Periodic Systems. IEEE Transactions on Nuclear
            Science, 20(3), 885–888. https://doi.org/10.1109/TNS.1973.4327279
        [2] Sagan, D., & Rubin, D. (1999). Linear analysis of coupled
            lattices. Physical Review Special Topics - Accelerators and Beams,
            2(7), 22–26. https://doi.org/10.1103/physrevstab.2.074001
        [3] C.J. Gardner, Some Useful Linear Coupling Approximations.
            C-A/AP/#101 Brookhaven Nat. Lab. (July 2003)

    Args:
        edteng_end (pyaccel.optics.EdwardsTengArray): EdwardsTeng parameters
            around the ring.

    Returns:
        min_tunesep (float) : estimative of minimum tune separation,
            Based on equation 85-87 of ref [3]:
            Assuming we are at the sum resonance, then T = 0.
            So we can write:
                mu1 = mu0 - minsep/2
                mu2 = mu0 + minsep/2
            where mu0 = (mu1 + mu2) / 2, and
                U = 2*sqrt(det(m+nbar))
            However, we know that
                U = 2*cos(mu1) - 2*cos(mu2)
                U = 2*cos(mu0-minsep/2) - 2*cos(mu0+minsep/2)
                U = 4*sin(mu0) * sin(minsep/2)
            which yields,
                sin(minsep/2) = sqrt(det(m+nbar))/sin(mu0)/2
        ratio (numpy.ndarray): estimative of invariant sharing ratio.
            Based on equation 216, 217 and 237 of ref [3].
            The ratio is not invariant along the ring.
            So the whole vector is returned.
            An average of this value could be used to estimate the ratio.

    """
    edt = edteng_end

    # ###### Estimative of emittance ratio #######

    # Equations 216 and 217 of ref [3]
    D2 = edt.beta2 * edt.W_22**2 + 2 * edt.alpha2 * edt.W_22 * edt.W_12
    D1 = edt.beta1 * edt.W_11**2 - 2 * edt.alpha1 * edt.W_11 * edt.W_12
    D2 += edt.gamma2 * edt.W_12**2
    D1 += edt.gamma1 * edt.W_12**2

    # Equation 237 of ref [3]
    ratio = 1 / edt.d**2 * _np.sqrt(D1 * D2 / edt.beta1 / edt.beta2)

    # # This second formula is based on equation 258 of ref [3] which is
    # # approximately valid for weak coupling:
    # dsqr = edt.d ** 2
    # ratio = (1-dsqr)/dsqr

    # ###### Estimate Minimum tune separation #####
    # from equations 85, 86 and 89 of ref [3]:
    edt = edt[-1]
    dsqr = edt.d * edt.d
    U = 2 * (_np.cos(edt.mu1) - _np.cos(edt.mu2))
    det_m_plus_nbar = U * U * dsqr * (1 - dsqr)

    mu0 = ((edt.mu1 % (2 * _np.pi)) + (edt.mu2 % (2 * _np.pi))) / 2
    min_tunesep = 2 * _np.arcsin(
        _np.sqrt(_np.abs(det_m_plus_nbar)) / _np.sin(mu0) / 2
    )
    min_tunesep /= 2 * _np.pi

    return min_tunesep, ratio


# 2-by-2 symplectic matrix
_S = _np.array([[0, 1], [-1, 0]])


def _trans(a):
    return a.transpose(0, 2, 1)


def _symplectic_transpose(a):
    """Definition in Eq 15 of ref [3]."""
    if len(a.shape) >= 3:
        return -_S @ _trans(a) @ _S
    else:
        return -_S @ a.T @ _S
