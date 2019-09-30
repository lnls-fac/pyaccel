
import sys as _sys
import math as _math
import numpy as _np
import mathphys as _mp
import trackcpp as _trackcpp
import pyaccel.lattice as _lattice
import pyaccel.tracking as _tracking
import pyaccel.accelerator as _accelerator
from pyaccel.utils import interactive as _interactive


class OpticsException(Exception):
    pass


class Twiss:

    def __init__(self, **kwargs):
        if 'twiss' in kwargs:
            if isinstance(kwargs['twiss'], _trackcpp.Twiss):
                copy = kwargs.get('copy', False)
                if copy:
                    self._t = _trackcpp.Twiss(kwargs['twiss'])
                else:
                    self._t = kwargs['twiss']
            elif isinstance(kwargs['twiss'], Twiss):
                copy = kwargs.get('copy', True)
                if copy:
                    self._t = _trackcpp.Twiss(kwargs['twiss']._t)
                else:
                    self._t = kwargs['twiss']._t
            else:
                raise TypeError(
                    'twiss must be a trackcpp.Twiss or a Twiss object.')
        else:
            self._t = _trackcpp.Twiss()

    def __eq__(self, other):
        if not isinstance(other, Twiss):
            return NotImplemented
        for attr in self._t.__swig_getmethods__:
            self_attr = getattr(self, attr)
            if isinstance(self_attr, _np.ndarray):
                if (self_attr != getattr(other, attr)).any():
                    return False
            else:
                if self_attr != getattr(other, attr):
                    return False
        return True

    @property
    def spos(self):
        return self._t.spos

    @spos.setter
    def spos(self, value):
        self._t.spos = value

    @property
    def rx(self):
        return self._t.co.rx

    @rx.setter
    def rx(self, value):
        self._t.co.rx = value

    @property
    def ry(self):
        return self._t.co.ry

    @ry.setter
    def ry(self, value):
        self._t.co.ry = value

    @property
    def px(self):
        return self._t.co.px

    @px.setter
    def px(self, value):
        self._t.co.px = value

    @property
    def py(self):
        return self._t.co.py

    @py.setter
    def py(self, value):
        self._t.co.py = value

    @property
    def de(self):
        return self._t.co.de

    @de.setter
    def de(self, value):
        self._t.co.de = value

    @property
    def dl(self):
        return self._t.co.dl

    @dl.setter
    def dl(self, value):
        self._t.co.dl = value

    @property
    def co(self):
        return _np.array([
            self.rx, self.px, self.ry, self.py, self.de, self.dl])

    @co.setter
    def co(self, value):
        self.rx, self.px = value[0], value[1]
        self.ry, self.py = value[2], value[3]
        self.de, self.dl = value[4], value[5]

    @property
    def betax(self):
        return self._t.betax

    @betax.setter
    def betax(self, value):
        self._t.betax = value

    @property
    def betay(self):
        return self._t.betay

    @betay.setter
    def betay(self, value):
        self._t.betay = value

    @property
    def alphax(self):
        return self._t.alphax

    @alphax.setter
    def alphax(self, value):
        self._t.alphax = value

    @property
    def alphay(self):
        return self._t.alphay

    @alphay.setter
    def alphay(self, value):
        self._t.alphay = value

    @property
    def mux(self):
        return self._t.mux

    @mux.setter
    def mux(self, value):
        self._t.mux = value

    @property
    def muy(self):
        return self._t.muy

    @muy.setter
    def muy(self, value):
        self._t.muy = value

    @property
    def etax(self):
        return self._t.etax[0]

    @etax.setter
    def etax(self, value):
        self._t.etax[0] = value

    @property
    def etay(self):
        return self._t.etay[0]

    @etay.setter
    def etay(self, value):
        self._t.etay[0] = value

    @property
    def etapx(self):
        return self._t.etax[1]

    @etapx.setter
    def etapx(self, value):
        self._t.etax[1] = value

    @property
    def etapy(self):
        return self._t.etay[1]

    @etapy.setter
    def etapy(self, value):
        self._t.etay[1] = value

    def __str__(self):
        r = ''
        r += 'spos          : ' + '{0:+10.3e}'.format(self.spos)
        fmt = '{0:+10.3e}, {1:+10.3e}'
        r += '\nrx, ry        : ' + fmt.format(self.rx, self.ry)
        r += '\npx, py        : ' + fmt.format(self.px, self.py)
        r += '\nde, dl        : ' + fmt.format(self.de, self.dl)
        r += '\nmux, muy      : ' + fmt.format(self.mux, self.muy)
        r += '\nbetax, betay  : ' + fmt.format(self.betax, self.betay)
        r += '\nalphax, alphay: ' + fmt.format(self.alphax, self.alphay)
        r += '\netax, etapx   : ' + fmt.format(self.etax, self.etapx)
        r += '\netay, etapy   : ' + fmt.format(self.etay, self.etapy)
        return r

    def make_dict(self):
        co = self.co
        beta = [self.betax, self.betay]
        alpha = [self.alphax, self.alphay]
        etax = [self.etax, self.etapx]
        etay = [self.etay, self.etapy]
        mu = [self.mux, self.muy]
        return {
            'co': co, 'beta': beta, 'alpha': alpha,
            'etax': etax, 'etay': etay, 'mu': mu}

    @staticmethod
    def make_new(*args, **kwrgs):
        """Build a Twiss object.
        """
        if args:
            if isinstance(args[0], dict):
                kwrgs = args[0]
        n = Twiss()
        n.co = kwrgs['co'] if 'co' in kwrgs else (0.0,)*6
        n.mux, n.muy = kwrgs['mu'] if 'mu' in kwrgs else (0.0, 0.0)
        n.betax, n.betay = kwrgs['beta'] if 'beta' in kwrgs else (0.0, 0.0)
        n.alphax, n.alphay = kwrgs['alpha'] if 'alpha' in kwrgs else (0.0, 0.0)
        n.etax, n.etapx = kwrgs['etax'] if 'etax' in kwrgs else (0.0, 0.0)
        n.etay, n.etapy = kwrgs['etay'] if 'etay' in kwrgs else (0.0, 0.0)
        return n


@_interactive
def calc_twiss(accelerator=None, init_twiss=None, fixed_point=None,
               indices='open', energy_offset=None):
    """Return Twiss parameters of uncoupled dynamics.

    Keyword arguments:
    accelerator   -- Accelerator object
    init_twiss    -- Twiss parameters at the start of first element
    fixed_point   -- 6D position at the start of first element
    indices       -- Open or closed
    energy_offset -- float denoting the energy deviation (used only for
                    periodic solutions).

    Returns:
    tw -- list of Twiss objects (closed orbit data is in the objects vector)
    m66 -- one-turn transfer matrix

    """
    if indices == 'open':
        closed_flag = False
    elif indices == 'closed':
        closed_flag = True
    else:
        raise OpticsException("invalid value for 'indices' in calc_twiss")

    _m66 = _trackcpp.Matrix()
    _twiss = _trackcpp.CppTwissVector()

    if init_twiss is not None:
        # as a transport line: uses init_twiss
        _init_twiss = init_twiss._t
        if fixed_point is None:
            _fixed_point = _init_twiss.co
        else:
            raise OpticsException(
                'arguments init_twiss and fixed_point are mutually exclusive')
    else:
        # as a periodic system: try to find periodic solution
        if accelerator.harmonic_number == 0:
            raise OpticsException(
                'Either harmonic number was not set or calc_twiss was'
                'invoked for transport line without initial twiss')

        if fixed_point is None:
            _closed_orbit = _trackcpp.CppDoublePosVector()
            _fixed_point_guess = _trackcpp.CppDoublePos()
            if energy_offset is not None:
                _fixed_point_guess.de = energy_offset

            if not accelerator.cavity_on and not accelerator.radiation_on:
                r = _trackcpp.track_findorbit4(
                    accelerator._accelerator, _closed_orbit,
                    _fixed_point_guess)
            elif not accelerator.cavity_on and accelerator.radiation_on:
                raise OpticsException(
                    'The radiation is on but the cavity is off')
            else:
                r = _trackcpp.track_findorbit6(
                    accelerator._accelerator, _closed_orbit,
                    _fixed_point_guess)

            if r > 0:
                raise _tracking.TrackingException(
                    _trackcpp.string_error_messages[r])
            _fixed_point = _closed_orbit[0]

        else:
            _fixed_point = _tracking._Numpy2CppDoublePos(fixed_point)
            if energy_offset is not None:
                _fixed_point.de = energy_offset

        _init_twiss = _trackcpp.Twiss()

    r = _trackcpp.calc_twiss(
        accelerator._accelerator, _fixed_point, _m66, _twiss,
        _init_twiss, closed_flag)
    if r > 0:
        raise OpticsException(_trackcpp.string_error_messages[r])

    twiss = TwissList(_twiss)
    m66 = _tracking._CppMatrix2Numpy(_m66)

    return twiss, m66


@_interactive
def calc_emittance_coupling(accelerator):
    # I copied the code below from:
    # http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    def fitEllipse(x, y):
        x = x[:, _np.newaxis]
        y = y[:, _np.newaxis]
        D = _np.hstack((x*x, x*y, y*y, x, y, _np.ones_like(x)))
        S = _np.dot(D.T, D)
        C = _np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = _np.linalg.eig(_np.linalg.solve(S, C))
        n = _np.argmax(_np.abs(E))
        a = V[:, n]
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1 = (b*b-a*c) * ((c-a) * _np.sqrt(1 + 4*b*b / ((a-c)*(a-c)))-(c+a))
        down2 = (b*b-a*c) * ((a-c) * _np.sqrt(1 + 4*b*b / ((a-c)*(a-c)))-(c+a))
        res1 = _np.sqrt(up/down1)
        res2 = _np.sqrt(up/down2)
        return _np.array([res1, res2])  # returns the axes of the ellipse

    acc = accelerator[:]
    acc.cavity_on = False
    acc.radiation_on = False

    orb = _tracking.find_orbit4(acc)
    rin = _np.array(
        [2e-5+orb[0], 0+orb[1], 1e-8+orb[2], 0+orb[3], 0, 0], dtype=float)
    rout, *_ = _tracking.ring_pass(
        acc, rin, nr_turns=100, turn_by_turn='closed', element_offset=0)
    r = _np.dstack([rin[None, :, None], rout])
    ax, bx = fitEllipse(r[0][0], r[0][1])
    ay, by = fitEllipse(r[0][2], r[0][3])
    return (ay*by) / (ax*bx)  # ey/ex


@_interactive
def get_rf_frequency(accelerator):
    """Return the frequency of the first RF cavity in the lattice"""
    for e in accelerator:
        if e.frequency != 0.0:
            return e.frequency
    else:
        raise OpticsException('no cavity element in the lattice')


@_interactive
def get_rf_voltage(accelerator):
    """Return the voltage of the first RF cavity in the lattice"""
    voltages = []
    for e in accelerator:
        if e.voltage != 0.0:
            voltages.append(e.voltage)
    if voltages:
        if len(voltages) == 1:
            return voltages[0]
        else:
            return voltages
    else:
        raise OpticsException('no cavity element in the lattice')


@_interactive
def get_revolution_period(accelerator):
    return accelerator.length/accelerator.velocity


@_interactive
def get_revolution_frequency(accelerator):
    return 1.0/get_revolution_period(accelerator)


@_interactive
def get_traces(accelerator=None, m66=None, closed_orbit=None):
    """Return traces of 6D one-turn transfer matrix"""
    if m66 is None:
        m66 = _tracking.find_m66(
            accelerator, indices='m66', closed_orbit=closed_orbit)
    trace_x = m66[0, 0] + m66[1, 1]
    trace_y = m66[2, 2] + m66[3, 3]
    trace_z = m66[4, 4] + m66[5, 5]
    return trace_x, trace_y, trace_z, m66, closed_orbit


@_interactive
def get_frac_tunes(accelerator=None, m66=None, closed_orbit=None,
                   coupled=False):
    """Return fractional tunes of the accelerator"""

    trace_x, trace_y, trace_z, m66, closed_orbit = get_traces(
        accelerator, m66=m66, closed_orbit=closed_orbit)
    tune_x = _math.acos(trace_x/2.0)/2.0/_math.pi
    tune_y = _math.acos(trace_y/2.0)/2.0/_math.pi
    tune_z = _math.acos(trace_z/2.0)/2.0/_math.pi
    if coupled:
        tunes = _np.log(_np.linalg.eigvals(m66))/2.0/_math.pi/1j
        tune_x = tunes[_np.argmin(abs(_np.sin(tunes.real)-_math.sin(tune_x)))]
        tune_y = tunes[_np.argmin(abs(_np.sin(tunes.real)-_math.sin(tune_y)))]
        tune_z = tunes[_np.argmin(abs(_np.sin(tunes.real)-_math.sin(tune_z)))]

    return tune_x, tune_y, tune_z, trace_x, trace_y, trace_z, m66, closed_orbit


@_interactive
def get_chromaticities(accelerator):
    raise OpticsException('not implemented')


@_interactive
def get_mcf(accelerator, order=1, energy_offset=None):
    """Return momentum compaction factor of the accelerator"""
    if energy_offset is None:
        energy_offset = _np.linspace(-1e-3, 1e-3, 11)

    accel = accelerator[:]
    _tracking.set_4d_tracking(accel)
    leng = accel.length

    dl = _np.zeros(_np.size(energy_offset))
    for i, ene in enumerate(energy_offset):
        fp = _tracking.find_orbit4(accel, ene)
        X0 = _np.concatenate([fp.flatten(), [ene, 0]])
        T, *_ = _tracking.ring_pass(accel, X0)
        dl[i] = T[0, 5]/leng

    polynom = _np.polyfit(energy_offset, dl, order)
    a = _np.fliplr([polynom])[0].tolist()
    a = a[1:]
    if len(a) == 1:
        a = a[0]
    return a


@_interactive
def get_beam_size(accelerator, coupling=0.0, closed_orbit=None, twiss=None,
                  indices='open'):
    """Return beamsizes (stddev) along ring"""

    # twiss parameters
    twi = twiss
    if twi is None:
        fpnt = closed_orbit if closed_orbit is None else closed_orbit[:, 0]
        twi, *_ = calc_twiss(accelerator, fixed_point=fpnt, indices=indices)

    betax, alphax, etax, etapx = twi.betax, twi.alphax, twi.etax, twi.etapx
    betay, alphay, etay, etapy = twi.betay, twi.alphay, twi.etay, twi.etapy

    gammax = (1.0 + alphax**2)/betax
    gammay = (1.0 + alphay**2)/betay

    # emittances and energy spread
    equi = EquilibriumParameters(accelerator)
    e0 = equi.emit0
    sigmae = equi.espread0
    ey = e0 * coupling / (1.0 + coupling)
    ex = e0 * 1 / (1.0 + coupling)

    # beamsizes per se
    sigmax = _np.sqrt(ex * betax + (sigmae * etax)**2)
    sigmay = _np.sqrt(ey * betay + (sigmae * etay)**2)
    sigmaxl = _np.sqrt(ex * gammax + (sigmae * etapx)**2)
    sigmayl = _np.sqrt(ey * gammay + (sigmae * etapy)**2)
    return sigmax, sigmay, sigmaxl, sigmayl, ex, ey, summary, twiss


@_interactive
def get_transverse_acceptance(accelerator, twiss=None, init_twiss=None,
                              fixed_point=None, energy_offset=0.0):
    """Return linear transverse horizontal and vertical physical acceptances"""

    m66 = None
    if twiss is None:
        twiss, m66 = calc_twiss(accelerator, init_twiss=init_twiss, fixed_point=fixed_point, indices='open')
        n = len(accelerator)
    else:
        if len(twiss) == len(accelerator):
            n = len(accelerator)
        elif len(twiss) == len(accelerator)+1:
            n = len(accelerator)+1
        else:
            raise OpticsException('Mismatch between size of accelerator and size of twiss object')

    # # Old get twiss
    # closed_orbit = _np.zeros((6,n))
    # closed_orbit[0,:], closed_orbit[2,:] = get_twiss(twiss, ('rx','ry'))
    # betax, betay, etax, etay = get_twiss(twiss, ('betax', 'betay', 'etax', 'etay'))

    closed_orbit = twiss.co
    betax, betay, etax, etay = twiss.betax, twiss.betay, twiss.etax, twiss.etay

    # physical apertures
    lattice = accelerator._accelerator.lattice
    hmax, vmax = _np.array([(lattice[i].hmax,lattice[i].vmax) for i in range(len(accelerator))]).transpose()
    if len(hmax) != n:
        hmax = _np.append(hmax, hmax[-1])
        vmax = _np.append(vmax, vmax[-1])

    # calcs local linear acceptances
    co_x, co_y = closed_orbit[(0,2),:]

    # calcs acceptance with beta at entrance of elements
    betax_sqrt, betay_sqrt = _np.sqrt(betax), _np.sqrt(betay)
    local_accepx_pos = (hmax - (co_x + etax * energy_offset)) / betax_sqrt
    local_accepx_neg = (hmax + (co_x + etax * energy_offset)) / betax_sqrt
    local_accepy_pos = (vmax - (co_y + etay * energy_offset)) / betay_sqrt
    local_accepy_neg = (vmax + (co_y + etay * energy_offset)) / betay_sqrt
    local_accepx_pos[local_accepx_pos < 0] = 0
    local_accepx_neg[local_accepx_neg < 0] = 0
    local_accepx_pos[local_accepy_pos < 0] = 0
    local_accepx_neg[local_accepy_neg < 0] = 0
    accepx_in = [min(local_accepx_pos[i],local_accepx_neg[i])**2 for i in range(n)]
    accepy_in = [min(local_accepy_pos[i],local_accepy_neg[i])**2 for i in range(n)]

    # calcs acceptance with beta at exit of elements
    betax_sqrt, betay_sqrt = _np.roll(betax_sqrt,-1), _np.roll(betay_sqrt,-1)
    local_accepx_pos = (hmax - (co_x + etax * energy_offset)) / betax_sqrt
    local_accepx_neg = (hmax + (co_x + etax * energy_offset)) / betax_sqrt
    local_accepy_pos = (vmax - (co_y + etay * energy_offset)) / betay_sqrt
    local_accepy_neg = (vmax + (co_y + etay * energy_offset)) / betay_sqrt
    local_accepx_pos[local_accepx_pos < 0] = 0
    local_accepx_neg[local_accepx_neg < 0] = 0
    local_accepx_pos[local_accepy_pos < 0] = 0
    local_accepx_neg[local_accepy_neg < 0] = 0
    accepx_out = [min(local_accepx_pos[i],local_accepx_neg[i])**2 for i in range(n)]
    accepy_out = [min(local_accepy_pos[i],local_accepy_neg[i])**2 for i in range(n)]

    accepx = [min(accepx_in[i],accepx_out[i]) for i in range(n)]
    accepy = [min(accepy_in[i],accepy_out[i]) for i in range(n)]

    if m66 is None:
        return accepx, accepy, twiss
    else:
        return accepx, accepy, twiss, m66


class EquilibriumParameters:

    def __init__(self, accelerator):
        self._acc = _accelerator.Accelerator()
        self._m66 = None
        self._twi = None
        self._alpha = 0.0
        self._integrals = _np.zeros(6)
        self._damping = _np.zeros(3)
        self._radiation_damping = _np.zeros(3)
        self.accelerator = accelerator

    def __str__(self):
        r = ''
        fmt = '{:<30s}: '
        fmtn = '{:.4g}'

        fmte = fmt + fmtn
        r += fmte.format('\nEnergy [GeV]', self.accelerator.energy*1e-9)

        ints = 'I1,I2,I3,I3a,I4,I5,I6'.split(',')
        r += '\n' + fmt.format(', '.join(ints))
        r += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'Jx,Jy,Je'.split(',')
        r += '\n' + fmt.format(', '.join(ints))
        r += ', '.join([fmtn.format(getattr(self, x)) for x in ints])

        ints = 'taux,tauy,taue'.split(',')
        r += '\n' + fmt.format(', '.join(ints) + ' [ms]')
        r += ', '.join([fmtn.format(1000*getattr(self, x)) for x in ints])

        r += fmte.format('\nmomentum compaction x 1e4', self.alpha*1e4)
        r += fmte.format('\nenergy loss [keV]', self.U0/1000)
        r += fmte.format('\novervoltage', self.overvoltage)
        r += fmte.format('\nsync phase [°]', self.syncphase*180/_math.pi)
        r += fmte.format('\nsync tune', self.synctune)
        r += fmte.format('\nnatural emittance [nm.rad]', self.emit0*1e9)
        r += fmte.format('\nnatural espread [%]', self.espread0*100)
        r += fmte.format('\nbunch length [mm]', self.bunch_length*1000)
        r += fmte.format('\nRF energy accep. [%]', self.rf_acceptance*100)
        return r

    @property
    def accelerator(self):
        return self._acc

    @accelerator.setter
    def accelerator(self, acc):
        if isinstance(acc, _accelerator.Accelerator):
            self._acc = acc
            self._calc_radiation_integrals()

    @property
    def twiss(self):
        return self._twi

    @property
    def m66(self):
        return self._m66

    @property
    def I1(self):
        return self._integrals[0]

    @property
    def I2(self):
        return self._integrals[1]

    @property
    def I3(self):
        return self._integrals[2]

    @property
    def I3a(self):
        return self._integrals[3]

    @property
    def I4(self):
        return self._integrals[4]

    @property
    def I5(self):
        return self._integrals[5]

    @property
    def I6(self):
        return self._integrals[6]

    @property
    def Jx(self):
        return 1.0 - self.I4/self.I2

    @property
    def Jy(self):
        return 1.0

    @property
    def Je(self):
        return 2.0 + self.I4/self.I2

    @property
    def alphax(self):
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jx / leng

    @property
    def alphay(self):
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Jy / leng

    @property
    def alphae(self):
        Ca = _mp.constants.Ca
        E0 = self._acc.energy / 1e9  # in GeV
        leng = self._acc.length
        return Ca * E0**3 * self.I2 * self.Je / leng

    @property
    def taux(self):
        return 1/self.alphax

    @property
    def tauy(self):
        return 1/self.alphay

    @property
    def taue(self):
        return 1/self.alphae

    @property
    def espread0(self):
        Cq = _mp.constants.Cq
        gamma = self._acc.gamma_factor
        return _math.sqrt(Cq * gamma**2 * self.I3 / (2*self.I2 + self.I4))

    @property
    def emit0(self):
        Cq = _mp.constants.Cq
        gamma = self._acc.gamma_factor
        return Cq * gamma**2 * self.I5 / (self.Jx*self.I2)

    @property
    def U0(self):
        E0 = self._acc.energy / 1e9  # in GeV
        rad_cgamma = _mp.constants.rad_cgamma
        return rad_cgamma/(2*_math.pi) * E0**4 * self.I2 * 1e9  # in GeV

    @property
    def overvoltage(self):
        v_cav = get_rf_voltage(self._acc)
        return v_cav/self.U0

    @property
    def syncphase(self):
        return _math.pi - _math.asin(1/self.overvoltage)

    @property
    def alpha(self):
        return self._alpha

    @property
    def etac(self):
        gamma = self._acc.gamma_factor
        return 1/(gamma*gamma) - self.alpha

    @property
    def synctune(self):
        E0 = self._acc.energy
        v_cav = get_rf_voltage(self._acc)
        harmon = self._acc.harmonic_number
        return _math.sqrt(
            self.etac*harmon*v_cav*_math.cos(self.syncphase)/(2*_math.pi*E0))

    @property
    def bunch_length(self):
        c = _mp.constants.light_speed
        beta = self._acc.beta_factor
        rev_freq = get_revolution_frequency(self._acc)

        bunlen = beta * c * abs(self.etac) * self.espread0
        bunlen /= 2*_math.pi * self.synctune * rev_freq
        return bunlen

    @property
    def rf_acceptance(self):
        E0 = self._acc.energy
        sph = self.syncphase
        V = get_rf_voltage(self._acc)
        ov = self.overvoltage
        h = self._acc.harmonic_number
        etac = self.etac

        eaccep2 = V * _math.sin(sph) / (_math.pi*h*abs(etac)*E0)
        eaccep2 *= 2 * (_math.sqrt(ov**2 - 1.0) - _math.acos(1.0/ov))
        return _math.sqrt(eaccep2)

    @staticmethod
    def calcH(beta, alpha, x, xl):
        gamma = (1 + alpha**2) / beta
        return beta*xl**2 + 2*alpha*x*xl + gamma*x**2

    def _calc_radiation_integrals(self):
        """Calculate radiation integrals for periodic systems"""

        acc = self._acc
        twi, m66 = calc_twiss(acc, indices='closed')
        self._twi = twi
        self._m66 = m66
        self._alpha = get_mcf(acc)

        spos = _lattice.find_spos(acc, indices='closed')
        etax, etapx, betax, alphax = twi.etax, twi.etapx, twi.betax, twi.alphax

        n = len(acc)
        angle, angle_in, angle_out, K = _np.zeros((4, n))
        for i in range(n):
            angle[i] = acc[i].angle
            angle_in[i] = acc[i].angle_in
            angle_out[i] = acc[i].angle_out
            K[i] = acc[i].K

        idx, *_ = _np.nonzero(angle)
        leng = spos[idx+1]-spos[idx]
        rho = leng/angle[idx]
        angle_in = angle_in[idx]
        angle_out = angle_out[idx]
        K = K[idx]
        etax_in, etax_out = etax[idx], etax[idx+1]
        etapx_in, etapx_out = etapx[idx], etapx[idx+1]
        betax_in, betax_out = betax[idx], betax[idx+1]
        alphax_in, alphax_out = alphax[idx], alphax[idx+1]

        H_in = self.calcH(betax_in, alphax_in, etax_in, etapx_in)
        H_out = self.calcH(betax_out, alphax_out, etax_in, etapx_out)

        etax_avg = (etax_in + etax_out) / 2
        H_avg = (H_in + H_out) / 2
        rho2, rho3 = rho**2, rho**3
        rho3abs = _np.abs(rho3)

        integrals = _np.zeros(7)
        integrals[0] = _np.dot(etax_avg/rho, leng)
        integrals[1] = _np.dot(1/rho2, leng)
        integrals[2] = _np.dot(1/rho3abs, leng)
        integrals[3] = _np.dot(1/rho3, leng)

        integrals[4] = _np.dot(etax_avg/rho3 * (1+2*rho2*K), leng)
        # for general wedge magnets:
        integrals[4] += sum((etax_in/rho2) * _np.tan(angle_in))
        integrals[4] += sum((etax_out/rho2) * _np.tan(angle_out))

        integrals[5] = _np.dot(H_avg / rho3abs, leng)
        integrals[6] = _np.dot((K*etax_avg)**2, leng)

        self._integrals = integrals


class TwissList(object):

    def __init__(self, twiss_list=None):
        """Read-only list of matrices.

        Keyword argument:
        twiss_list -- trackcpp Twiss vector (default: None)
        """
        # TEST!
        if twiss_list is not None:
            if isinstance(twiss_list, _trackcpp.CppTwissVector):
                self._tl = twiss_list
            else:
                raise TrackingException('invalid Twiss vector')
        else:
            self._tl = _trackcpp.CppTwissVector()
        self._ptl = [self._tl[i] for i in range(len(self._tl))]

    def __len__(self):
        return len(self._tl)

    def __getitem__(self, index):
        if isinstance(index,(int, _np.int_)):
            return Twiss(twiss=self._tl[index])
        elif isinstance(index, (list,tuple,_np.ndarray)) and all(isinstance(x, (int, _np.int_)) for x in index):
            tl = _trackcpp.CppTwissVector()
            for i in index:
                tl.append(self._tl[i])
            return TwissList(twiss_list = tl)
        elif isinstance(index, slice):
            return TwissList(twiss_list = self._tl[index])
        else:
            raise TypeError('invalid index')

    def append(self, value):
        if isinstance(value, _trackcpp.Twiss):
            self._tl.append(value)
            self._ptl.append(value)
        elif isinstance(value, Twiss):
            self._tl.append(value._t)
            self._ptl.append(value._t)
        elif self._is_list_of_lists(value):
            t = _trackcpp.Twiss()
            for line in value:
                t.append(line)
            self._tl.append(t)
            self._ptl.append(t)
        else:
            raise TrackingException('can only append twiss-like objects')

    def _is_list_of_lists(self, value):
        valid_types = (list, tuple)
        if not isinstance(value, valid_types):
            return False
        for line in value:
            if not isinstance(line, valid_types):
                return False
        return True

    @property
    def spos(self):
        spos = _np.array([float(self._ptl[i].spos) for i in range(len(self._ptl))])
        return spos if len(spos) > 1 else spos[0]

    @property
    def betax(self):
        betax = _np.array([float(self._ptl[i].betax) for i in range(len(self._ptl))])
        return betax if len(betax) > 1 else betax[0]

    @property
    def betay(self):
        betay = _np.array([float(self._ptl[i].betay) for i in range(len(self._ptl))])
        return betay if len(betay) > 1 else betay[0]

    @property
    def alphax(self):
        alphax = _np.array([float(self._ptl[i].alphax) for i in range(len(self._ptl))])
        return alphax if len(alphax) > 1 else alphax[0]

    @property
    def alphay(self):
        alphay = _np.array([float(self._ptl[i].alphay) for i in range(len(self._ptl))])
        return alphay if len(alphay) > 1 else alphay[0]

    @property
    def mux(self):
        mux = _np.array([float(self._ptl[i].mux) for i in range(len(self._ptl))])
        return mux if len(mux) > 1 else mux[0]

    @property
    def muy(self):
        muy = _np.array([float(self._ptl[i].muy) for i in range(len(self._ptl))])
        return muy if len(muy) > 1 else muy[0]

    @property
    def etax(self):
        etax = _np.array([float(self._ptl[i].etax[0]) for i in range(len(self._ptl))])
        return etax if len(etax) > 1 else etax[0]

    @property
    def etay(self):
        etay = _np.array([float(self._ptl[i].etay[0]) for i in range(len(self._ptl))])
        return etay if len(etay) > 1 else etay[0]

    @property
    def etapx(self):
        etapx = _np.array([float(self._ptl[i].etax[1]) for i in range(len(self._ptl))])
        return etapx if len(etapx) > 1 else etapx[0]

    @property
    def etapy(self):
        etapy = _np.array([float(self._ptl[i].etay[1]) for i in range(len(self._ptl))])
        return etapy if len(etapy) > 1 else etapy[0]

    @property
    def rx(self):
        res = _np.array([float(ptl.co.rx) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def ry(self):
        res = _np.array([float(ptl.co.ry) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def px(self):
        res = _np.array([float(ptl.co.px) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def py(self):
        res = _np.array([float(ptl.co.py) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def de(self):
        res = _np.array([float(ptl.co.de) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def dl(self):
        res = _np.array([float(ptl.co.dl) for ptl in self._ptl])
        return res if len(res) > 1 else res[0]

    @property
    def co(self):
        co = [self._ptl[i].co for i in range(len(self._ptl))]
        co = [[co[i].rx, co[i].px, co[i].ry, co[i].py, co[i].de, co[i].dl] for i in range(len(co))]
        co = _np.transpose(_np.array(co))
        return co if len(co[0,:]) > 1 else co[:,0]


''' deprecated: graphics module needs to be updated before get_twiss is deleted '''
@_interactive
def get_twiss(twiss_list, attribute_list):
    """Build a matrix with Twiss data from a list of Twiss objects.

    Accepts a list of Twiss objects and returns a matrix with Twiss data, one line for
    each Twiss parameter defined in 'attributes_list'.

    Keyword arguments:
    twiss_list -- List with Twiss objects
    attributes_list -- List of strings with Twiss attributes to be stored in twiss matrix

    Returns:
    m -- Matrix with Twiss data. Can also be thought of a single column of
         Twiss parameter vectors:
            betax, betay = get_twiss(twiss, ('betax','betay'))
    """
    if isinstance(attribute_list, str):
        attribute_list = (attribute_list,)
    values = _np.zeros((len(attribute_list),len(twiss_list)))
    for i in range(len(twiss_list)):
        for j in range(len(attribute_list)):
            values[j,i] = getattr(twiss_list[i], attribute_list[j])
    if values.shape[0] == 1:
        return values[0,:]
    else:
        return values
