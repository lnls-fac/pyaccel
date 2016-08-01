
import sys as _sys
import math as _math
import numpy as _np
import mathphys as _mp
import pyaccel.lattice as _lattice
import pyaccel.tracking as _tracking
import trackcpp as _trackcpp
from pyaccel.utils import interactive as _interactive


class OpticsException(Exception):
    pass


class Twiss:

    def __init__(self, **kwargs):
        if 'twiss' in kwargs:
            if isinstance(kwargs['twiss'],_trackcpp.Twiss):
                copy = kwargs.get('copy',False)
                if copy:
                    self._t = _trackcpp.Twiss(kwargs['twiss'])
                else:
                    self._t = kwargs['twiss']
            elif isinstance(kwargs['twiss'],Twiss):
                copy = kwargs.get('copy',True)
                if copy:
                    self._t = _trackcpp.Twiss(kwargs['twiss']._t)
                else:
                    self._t = kwargs['twiss']._t
            else:
                raise TypeError('twiss must be a trackcpp.Twiss or a Twiss object.')
        else:
            self._t = _trackcpp.Twiss()

    def __eq__(self,other):
        if not isinstance(other,Twiss): return NotImplemented
        for attr in self._t.__swig_getmethods__:
            self_attr = getattr(self,attr)
            if isinstance(self_attr,_np.ndarray):
                if (self_attr != getattr(other,attr)).any():
                    return False
            else:
                if self_attr != getattr(other,attr):
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
        return _np.array([self.rx, self.px, self.ry, self.py, self.de, self.dl])

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
        r +=   'spos          : ' + '{0:+10.3e}'.format(self.spos)
        r += '\nrx, ry        : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.rx, self.ry)
        r += '\npx, py        : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.px, self.py)
        r += '\nde, dl        : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.de, self.dl)
        r += '\nmux, muy      : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.mux, self.muy)
        r += '\nbetax, betay  : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.betax, self.betay)
        r += '\nalphax, alphay: ' + '{0:+10.3e}, {1:+10.3e}'.format(self.alphax, self.alphay)
        r += '\netax, etapx   : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.etax, self.etapx)
        r += '\netay, etapy   : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.etay, self.etapy)
        return r

    def make_dict(self):
        co    =  self.co
        beta  = [self.betax, self.betay]
        alpha = [self.alphax, self.alphay]
        etax  = [self.etax, self.etapx]
        etay  = [self.etay, self.etapy]
        mu    = [self.mux, self.muy]
        _dict = {'co': co, 'beta': beta, 'alpha': alpha, 'etax': etax, 'etay': etay, 'mu': mu}
        return _dict

    @staticmethod
    def make_new(*args, **kwargs):
        """Build a Twiss object.
        """
        if args:
            if isinstance(args[0], dict):
                kwargs = args[0]
        n = Twiss()
        n.co = kwargs['co'] if 'co' in kwargs else (0.0,)*6
        n.mux,    n.muy    = kwargs['mu']    if 'mu'    in kwargs else (0.0, 0.0)
        n.betax,  n.betay  = kwargs['beta']  if 'beta'  in kwargs else (0.0, 0.0)
        n.alphax, n.alphay = kwargs['alpha'] if 'alpha' in kwargs else (0.0, 0.0)
        n.etax,   n.etapx  = kwargs['etax']  if 'etax'  in kwargs else (0.0, 0.0)
        n.etay,   n.etapy  = kwargs['etay']  if 'etay'  in kwargs else (0.0, 0.0)
        return n


@_interactive
def calc_twiss(accelerator=None, init_twiss=None, fixed_point=None, indices = 'open', energy_offset=None):
    """Return Twiss parameters of uncoupled dynamics.

    Keyword arguments:
    accelerator   -- Accelerator object
    init_twiss    -- Twiss parameters at the start of first element
    fixed_point   -- 6D position at the start of first element
    indices       -- Open or closed
    energy_offset -- float denoting the energy deviation (used only for periodic
                     solutions).

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

    _m66   = _trackcpp.Matrix()
    _twiss = _trackcpp.CppTwissVector()

    if init_twiss is not None:
        ''' as a transport line: uses init_twiss '''
        _init_twiss = init_twiss._t
        if fixed_point is None:
            _fixed_point = _init_twiss.co
        else:
            raise OpticsException('arguments init_twiss and fixed_point are mutually exclusive')
        r = _trackcpp.calc_twiss(accelerator._accelerator, _fixed_point, _m66, _twiss, _init_twiss, closed_flag)

    else:
        ''' as a periodic system: try to find periodic solution '''
        if accelerator.harmonic_number == 0:
            raise OpticsException('Either harmonic number was not set or calc_twiss was'
                'invoked for transport line without initial twiss')

        if fixed_point is None:
            _closed_orbit = _trackcpp.CppDoublePosVector()
            _fixed_point_guess = _trackcpp.CppDoublePos()
            if energy_offset is not None: _fixed_point_guess.de = energy_offset

            if not accelerator.cavity_on and not accelerator.radiation_on:
                r = _trackcpp.track_findorbit4(accelerator._accelerator, _closed_orbit, _fixed_point_guess)
            elif not accelerator.cavity_on and accelerator.radiation_on:
                raise OpticsException('The radiation is on but the cavity is off')
            else:
                r = _trackcpp.track_findorbit6(accelerator._accelerator, _closed_orbit, _fixed_point_guess)

            if r > 0:
                raise _tracking.TrackingException(_trackcpp.string_error_messages[r])
            _fixed_point = _closed_orbit[0]

        else:
            _fixed_point = _tracking._Numpy2CppDoublePos(fixed_point)
            if energy_offset is not None: _fixed_point.de = energy_offset

        r = _trackcpp.calc_twiss(accelerator._accelerator, _fixed_point, _m66, _twiss)

    if r > 0:
        raise OpticsException(_trackcpp.string_error_messages[r])

    twiss = TwissList(_twiss)
    m66 = _tracking._CppMatrix2Numpy(_m66)

    return twiss, m66


@_interactive
def calc_emittance_coupling(accelerator):
    # I copied the code below from:
    # http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    def fitEllipse(x,y):
        x = x[:,_np.newaxis]
        y = y[:,_np.newaxis]
        D = _np.hstack((x*x, x*y, y*y, x, y, _np.ones_like(x)))
        S = _np.dot(D.T,D)
        C = _np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V = _np.linalg.eig(_np.linalg.solve(S, C))
        n = _np.argmax(_np.abs(E))
        a = V[:,n]
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up    = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1 = (b*b-a*c)*( (c-a)*_np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2 = (b*b-a*c)*( (a-c)*_np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1 = _np.sqrt(up/down1)
        res2 = _np.sqrt(up/down2)
        return _np.array([res1, res2]) #returns the axes of the ellipse

    acc = accelerator[:]
    acc.cavity_on    = False
    acc.radiation_on = False

    orb = _tracking.find_orbit4(acc)
    rin = _np.array([2e-5+orb[0],0+orb[1],1e-8+orb[2],0+orb[3],0,0],dtype=float)
    rout, *_ = _tracking.ring_pass(acc, rin, nr_turns=100, turn_by_turn='closed',element_offset = 0)
    r = _np.dstack([rin[None,:,None],rout])
    ax,bx = fitEllipse(r[0][0],r[0][1])
    ay,by = fitEllipse(r[0][2],r[0][3])
    return (ay*by) / (ax*bx) # ey/ex


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
        m66 = _tracking.find_m66(accelerator,
                                indices = 'm66', closed_orbit = closed_orbit)
    trace_x = m66[0,0] + m66[1,1]
    trace_y = m66[2,2] + m66[3,3]
    trace_z = m66[4,4] + m66[5,5]
    return trace_x, trace_y, trace_z, m66, closed_orbit


@_interactive
def get_frac_tunes(accelerator=None, m66=None, closed_orbit=None, coupled=False):
    """Return fractional tunes of the accelerator"""

    trace_x, trace_y, trace_z, m66, closed_orbit = get_traces(accelerator,
                                                   m66 = m66,
                                                   closed_orbit = closed_orbit)
    tune_x = _math.acos(trace_x/2.0)/2.0/_math.pi
    tune_y = _math.acos(trace_y/2.0)/2.0/_math.pi
    tune_z = _math.acos(trace_z/2.0)/2.0/_math.pi
    if coupled:
        tunes = _np.log(_np.linalg.eigvals(m66))/2.0/_math.pi/1j
        tune_x = tunes[_np.argmin(abs(_np.sin(tunes.real) - _math.sin(tune_x)))]
        tune_y = tunes[_np.argmin(abs(_np.sin(tunes.real) - _math.sin(tune_y)))]
        tune_z = tunes[_np.argmin(abs(_np.sin(tunes.real) - _math.sin(tune_z)))]

    return tune_x, tune_y, tune_z, trace_x, trace_y, trace_z, m66, closed_orbit


@_interactive
def get_chromaticities(accelerator):
    raise OpticsException('not implemented')


@_interactive
def get_mcf(accelerator, order=1, energy_offset=None):
    """Return momentum compaction factor of the accelerator"""
    if energy_offset is None:
        energy_offset = _np.linspace(-1e-3,1e-3,11)

    accel=accelerator[:]
    _tracking.set_4d_tracking(accel)
    ring_length = _lattice.length(accel)

    dl = _np.zeros(_np.size(energy_offset))
    for i in range(len(energy_offset)):
        fp = _tracking.find_orbit4(accel,energy_offset[i])
        X0 = _np.concatenate([fp,[energy_offset[i],0]]).tolist()
        T = _tracking.ring_pass(accel,X0)
        dl[i] = T[0][5]/ring_length

    polynom = _np.polyfit(energy_offset,dl,order)
    a = _np.fliplr([polynom])[0].tolist()
    a = a[1:]
    if len(a) == 1:
        a=a[0]
    return a


@_interactive
def get_radiation_integrals(accelerator,
                          twiss=None,
                          m66=None,
                          closed_orbit=None):
    """Calculate radiation integrals for periodic systems"""

    if twiss is None or m66 is None:
        fixed_point = closed_orbit if closed_orbit is None else closed_orbit[:,0]
        twiss, m66 = calc_twiss(accelerator, fixed_point=fixed_point)

    spos = _lattice.find_spos(accelerator)

    # # Old get twiss
    # etax,etapx,betax,alphax = twiss,('etax','etapx','betax','alphax'))

    etax, etapx, betax, alphax = twiss.etax, twiss.etapx, twiss.betax, twiss.alphax

    if len(spos) != len(accelerator) + 1:
        spos = _np.resize(spos,len(accelerator)+1); spos[-1] = spos[-2] + accelerator[-1].length
        etax = _np.resize(etax,len(accelerator)+1); etax[-1] = etax[0]
        etapx = _np.resize(etapx,len(accelerator)+1); etapx[-1] = etapx[0]
        betax = _np.resize(betax,len(accelerator)+1); betax[-1] = betax[0]
        alphax = _np.resize(alphax,len(accelerator)+1); alphax[-1] = alphax[0]
    gammax = (1+alphax**2)/betax
    n = len(accelerator)
    angle, angle_in, angle_out, K = _np.zeros((4,n))
    for i in range(n):
        angle[i] = accelerator._accelerator.lattice[i].angle
        angle_in[i] = accelerator._accelerator.lattice[i].angle_in
        angle_out[i] = accelerator._accelerator.lattice[i].angle_out
        K[i] = accelerator._accelerator.lattice[i].polynom_b[1]
    idx, *_ = _np.nonzero(angle)
    leng = spos[idx+1]-spos[idx]
    rho  = leng/angle[idx]
    angle_in = angle_in[idx]
    angle_out = angle_out[idx]
    K = K[idx]
    etax_in, etax_out = etax[idx], etax[idx+1]
    etapx_in, etapx_out = etapx[idx], etapx[idx+1]
    betax_in, betax_out = betax[idx], betax[idx+1]
    alphax_in, alphax_out = alphax[idx], alphax[idx+1]
    gammax_in, gammax_out = gammax[idx], gammax[idx+1]
    H_in = betax_in*etapx_in**2 + 2*alphax_in*etax_in*etapx_in+gammax_in*etax_in**2
    H_out = betax_out*etapx_out**2 + 2*alphax_out*etax_out*etapx_out+gammax_out*etax_out**2

    etax_avg = 0.5*(etax_in+etax_out)
    rho2, rho3 = rho**2, rho**3
    rho3abs = _np.abs(rho3)

    integrals = [0.0]*6
    integrals[0] = _np.dot(etax_avg/rho, leng)
    integrals[1] = _np.dot(1/rho2, leng)
    integrals[2] = _np.dot(1/rho3abs, leng)
    integrals[3] = sum((etax_in/rho2)*_np.tan(angle_in)) + \
                   sum((etax_out/rho2)*_np.tan(angle_out)) + \
                   _np.dot((etax_avg/rho3)*(1+2*rho2*K), leng)
    integrals[4] = _np.dot(0.5*(H_in+H_out)/rho3abs, leng)
    integrals[5] = _np.dot((K*etax_avg)**2, leng)

    return integrals, twiss, m66


@_interactive
def get_natural_energy_spread(accelerator):
    Cq = _mp.constants.Cq
    gamma = accelerator.gamma_factor
    integrals, *_ = get_radiation_integrals(accelerator)
    natural_energy_spread = _math.sqrt( Cq*(gamma**2)*integrals[2]/(2*integrals[1] + integrals[3]))
    return natural_energy_spread


@_interactive
def get_natural_emittance(accelerator):
    Cq = _mp.constants.Cq
    gamma = accelerator.gamma_factor
    integrals, *_ = get_radiation_integrals(accelerator)

    damping = _np.zeros(3)
    damping[0] = 1.0 - integrals[3]/integrals[1]
    damping[1] = 1.0
    damping[2] = 2.0 + integrals[3]/integrals[1]

    natural_emittance = Cq*(gamma**2)*integrals[4]/(damping[0]*integrals[1])
    return natural_emittance


@_interactive
def get_natural_bunch_length(accelerator):
    c = _mp.constants.light_speed
    rad_cgamma = _mp.constants.rad_cgamma
    e0 = accelerator.energy
    gamma = accelerator.gamma_factor
    beta = accelerator.beta_factor
    harmon = accelerator.harmonic_number

    integrals, *_ = get_radiation_integrals(accelerator)
    rev_freq = get_revolution_frequency(accelerator)
    compaction_factor = get_mcf(accelerator)

    etac = gamma**(-2) - compaction_factor

    freq = get_rf_frequency(accelerator)
    v_cav = get_rf_voltage(accelerator)
    radiation = rad_cgamma*((e0/1e9)**4)*integrals[1]/(2*_math.pi)*1e9
    overvoltage = v_cav/radiation

    syncphase = _math.pi - _math.asin(1/overvoltage)
    synctune = _math.sqrt((etac * harmon * v_cav * _math.cos(syncphase))/(2*_math.pi*e0))
    natural_energy_spread = get_natural_energy_spread(accelerator)
    bunch_length = beta* c *abs(etac)* natural_energy_spread /( synctune * rev_freq *2*_math.pi)
    return bunch_length


@_interactive
def get_equilibrium_parameters(accelerator,
                             twiss=None,
                             m66=None,
                             closed_orbit=None):

    c = _mp.constants.light_speed
    Cq = _mp.constants.Cq
    Ca = _mp.constants.Ca
    rad_cgamma = _mp.constants.rad_cgamma

    e0 = accelerator.energy
    gamma = accelerator.gamma_factor
    beta = accelerator.beta_factor
    harmon = accelerator.harmonic_number
    circumference = accelerator.length
    rev_time = circumference / accelerator.velocity
    rev_freq = get_revolution_frequency(accelerator)

    compaction_factor = get_mcf(accelerator)
    etac = gamma**(-2) - compaction_factor

    integrals, *args = get_radiation_integrals(accelerator,twiss,m66,closed_orbit)

    damping = _np.zeros(3)
    damping[0] = 1.0 - integrals[3]/integrals[1]
    damping[1] = 1.0
    damping[2] = 2.0 + integrals[3]/integrals[1]

    radiation_damping = _np.zeros(3)
    radiation_damping[0] = 1.0/(Ca*((e0/1e9)**3)*integrals[1]*damping[0]/circumference)
    radiation_damping[1] = 1.0/(Ca*((e0/1e9)**3)*integrals[1]*damping[1]/circumference)
    radiation_damping[2] = 1.0/(Ca*((e0/1e9)**3)*integrals[1]*damping[2]/circumference)

    radiation = rad_cgamma*((e0/1e9)**4)*integrals[1]/(2*_math.pi)*1e9
    natural_energy_spread = _math.sqrt( Cq*(gamma**2)*integrals[2]/(2*integrals[1] + integrals[3]))
    natural_emittance = Cq*(gamma**2)*integrals[4]/(damping[0]*integrals[1])

    freq = get_rf_frequency(accelerator)
    v_cav = get_rf_voltage(accelerator)
    overvoltage = v_cav/radiation

    syncphase = _math.pi - _math.asin(1.0/overvoltage)
    synctune = _math.sqrt((etac * harmon * v_cav * _math.cos(syncphase))/(2*_math.pi*e0))
    rf_energy_acceptance = _math.sqrt(v_cav*_math.sin(syncphase)*2*(_math.sqrt((overvoltage**2)-1.0)
                        - _math.acos(1.0/overvoltage))/(_math.pi*harmon*abs(etac)*e0))
    bunch_length = beta* c *abs(etac)* natural_energy_spread /( synctune * rev_freq *2*_math.pi)

    summary=dict(compaction_factor = compaction_factor, radiation_integrals = integrals, damping_numbers = damping,
        damping_times = radiation_damping, natural_energy_spread = natural_energy_spread, etac = etac,
        natural_emittance = natural_emittance, overvoltage = overvoltage, syncphase = syncphase,
        synctune = synctune, rf_energy_acceptance = rf_energy_acceptance, bunch_length = bunch_length)

    return [summary, integrals] + args


@_interactive
def get_beam_size(accelerator, coupling=0.0, closed_orbit=None, indices='open'):
    """Return beamsizes (stddev) along ring"""

    # twiss parameters
    fixed_point = closed_orbit if closed_orbit is None else closed_orbit[:,0]
    twiss, *_ = calc_twiss(accelerator, fixed_point=fixed_point, indices=indices)

    # Old get twiss
    # betax, alphax, etax, etapx = get_twiss(twiss, ('betax','alphax','etax','etapx'))
    # betay, alphay, etay, etapy = get_twiss(twiss, ('betay','alphay','etay','etapy'))

    betax, alphax, etax, etapx = twiss.betax, twiss.alphax, twiss.etax, twiss.etapx
    betay, alphay, etay, etapy = twiss.betay, twiss.alphay, twiss.etay, twiss.etapy

    gammax = (1.0 + alphax**2)/betax
    gammay = (1.0 + alphay**2)/betay
    # emittances and energy spread
    summary, *_ = get_equilibrium_parameters(accelerator)
    e0 = summary['natural_emittance']
    sigmae = summary['natural_energy_spread']
    ey = e0 * coupling / (1.0 + coupling)
    ex = e0 * 1 / (1.0 + coupling)
    # beamsizes per se
    sigmax  = _np.sqrt(ex * betax + (sigmae * etax)**2)
    sigmay  = _np.sqrt(ey * betay + (sigmae * etay)**2)
    sigmaxl = _np.sqrt(ex * gammax + (sigmae * etapx)**2)
    sigmayl = _np.sqrt(ey * gammay + (sigmae * etapy)**2)
    return sigmax, sigmay, sigmaxl, sigmayl, ex, ey, summary, twiss


@_interactive
def get_transverse_acceptance(accelerator, twiss=None, init_twiss=None, fixed_point=None, energy_offset=0.0):
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
