
import time as _time
import copy as _copy
import math as _math
import numpy as _np
import mathphys as _mp
import pyaccel.lattice as _lattice
import pyaccel.tracking as _tracking
from pyaccel.utils import interactive as _interactive


class OpticsException(Exception):
    pass


class Twiss:
    def __init__(self):
        self.spos = 0
        self.rx, self.px  = 0, 0
        self.ry, self.py  = 0, 0
        self.de, self.dl  = 0, 0
        self.etax, self.etapx = 0, 0
        self.etay, self.etapy = 0, 0
        self.mux, self.betax, self.alphax = 0, None, None
        self.muy, self.betay, self.alphay = 0, None, None

    def __str__(self):
        r = ''
        r += 'spos          : ' + '{0:+10.3e}'.format(self.spos) + '\n'
        r += 'rx, ry        : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.rx, self.ry) + '\n'
        r += 'px, py        : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.px, self.py) + '\n'
        r += 'de, dl        : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.de, self.dl) + '\n'
        r += 'mux, muy      : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.mux, self.muy) + '\n'
        r += 'betax, betay  : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.betax, self.betay) + '\n'
        r += 'alphax, alphay: ' + '{0:+10.3e}, {1:+10.3e}'.format(self.alphax, self.alphay) + '\n'
        r += 'etax, etay    : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.etax, self.etay) + '\n'
        r += 'etapx, etapy  : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.etapx, self.etapy) + '\n'
        return r

    def make_copy(self):
        n = Twiss()
        n.spos = self.spos
        n.rx, n.px = self.rx, self.px
        n.ry, n.py = self.ry, self.py
        n.de, n.dl = self.de, self.dl
        n.etax, n.etapx = self.etax, self.etapx
        n.etay, n.etapy = self.etay, self.etapy
        n.mux, n.betax, n.alphax = self.mux, self.betax, self.alphax
        n.muy, n.betay, n.alphay = self.muy, self.betay, self.alphay
        return n

    @staticmethod
    def make_new(spos=0.0, fixed_point=None, mu=None, beta=None, alpha=None, eta=None, etal=None):
        n = Twiss()
        n.spos = spos
        if fixed_point is None:
            n.rx, n.px, n.ry, n.py, n.de, n.dl = (0.0,) * 6
        else:
            n.rx, n.px, n.ry, n.py, n.de, n.dl = fixed_point
        if mu is None:
            n.mux, n.muy = 0.0, 0.0
        else:
            n.mux, n.muy = mu
        if beta is None:
            n.betax, n.betay = None, None
        else:
            n.betax, n.betay = beta
        if alpha is None:
            n.alphax, n.alphay = 0.0, 0.0
        else:
            n.alphax, n.alphay = alpha
        if eta is None:
            n.etax, n.etay = 0.0, 0.0
        else:
            n.etax, n.etay = eta
        if etal is None:
            n.etalx, n.etaly = 0.0, 0.0
        else:
            n.etalx, n.etaly = etal
        return n

    @property
    def fixed_point(self):
        rx, px = self.rx, self.px
        ry, py = self.ry, self.py
        de, dl = self.de, self.dl
        fixed_point = [rx, px, ry, py, de, dl]
        return fixed_point

    @fixed_point.setter
    def fixed_point(self, value):
        self.rx, self.px  = value[0], value[1]
        self.ry, self.py  = value[2], value[3]
        self.de, self.dl  = value[4], value[5]

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

@_interactive
def calc_twiss(accelerator=None, init_twiss=None, fixed_point=None, indices = 'open'):
    """Return Twiss parameters of uncoupled dynamics.

    Keyword arguments:
    accelerator -- Accelerator object
    init_twiss  -- Twiss parameters at the start of first element
    fixed_point -- 6D position at the start of first element
    indices     -- Open or closed
    """

    if indices == 'open':
        length = len(accelerator)
    elif indices == 'closed':
        length = len(accelerator)+1
    else:
        raise OpticsException("invalid value for 'indices' in calc_twiss")

    if init_twiss is not None:
        ''' as a transport line: uses init_twiss '''
        if fixed_point is None:
            fixed_point = init_twiss.fixed_point
        else:
            raise OpticsException('arguments init_twiss and fixed_orbit are mutually exclusive')
        closed_orbit, *_ = _tracking.linepass(accelerator, particles=list(fixed_point), indices = indices)
        m66, transfer_matrices, *_ = _tracking.findm66(accelerator, closed_orbit = closed_orbit)
        mx, my = m66[0:2,0:2], m66[2:4,2:4]
        t = init_twiss
        t.etax = _np.array([[t.etax], [t.etapx]])
        t.etay = _np.array([[t.etay], [t.etapy]])
    else:
        ''' as a periodic system: try to find periodic solution '''
        if fixed_point is None:
            if not accelerator.cavity_on and not accelerator.radiation_on:
                if accelerator.harmonic_number == 0:
                    raise OpticsException('Either harmonic number was not set or calc_twiss was'
                    'invoked for transport line without initial twiss')
                closed_orbit = _np.zeros((6,length))
                closed_orbit[:4,:] = _tracking.findorbit4(accelerator, indices=indices)
            else:
                closed_orbit = _tracking.findorbit6(accelerator, indices=indices)
        else:
            closed_orbit, *_ = _tracking.linepass(accelerator, particles=list(fixed_point), indices=indices)

        ''' calcs twiss at first element '''
        m66, transfer_matrices, *_ = _tracking.findm66(accelerator, closed_orbit=closed_orbit)
        mx, my = m66[0:2,0:2], m66[2:4,2:4] # decoupled transfer matrices
        trace_x, trace_y, *_ = get_traces(accelerator, m66 = m66, closed_orbit=closed_orbit)
        if not (-2.0 < trace_x < 2.0):
            raise OpticsException('horizontal dynamics is unstable')
        if not (-2.0 < trace_y < 2.0):
            raise OpticsException('vertical dynamics is unstable')
        sin_nux = _math.copysign(1,mx[0,1]) * _math.sqrt(-mx[0,1] * mx[1,0] - ((mx[0,0] - mx[1,1])**2)/4);
        sin_nuy = _math.copysign(1,my[0,1]) * _math.sqrt(-my[0,1] * my[1,0] - ((my[0,0] - my[1,1])**2)/4);
        fp = closed_orbit[:,0]
        t = Twiss()
        t.spos = 0
        t.rx, t.px = fp[0], fp[1]
        t.ry, t.py = fp[2], fp[3]
        t.de, t.dl = fp[4], fp[5]
        t.alphax = (mx[0,0] - mx[1,1]) / 2 / sin_nux
        t.betax  = mx[0,1] / sin_nux
        t.alphay = (my[0,0] - my[1,1]) / 2 / sin_nuy
        t.betay  = my[0,1] / sin_nuy
        ''' dispersion function based on eta = (1 - M)^(-1) D '''
        Dx = _np.array([[m66[0,4]],[m66[1,4]]])
        Dy = _np.array([[m66[2,4]],[m66[3,4]]])
        t.etax = _np.linalg.solve(_np.eye(2,2) - mx, Dx)
        t.etay = _np.linalg.solve(_np.eye(2,2) - my, Dy)

    ''' propagates twiss through line '''
    tw = [t]
    m_previous = _np.eye(6,6)
    for i in range(1, length):
        m = transfer_matrices[i-1]
        mx, my = m[0:2,0:2], m[2:4,2:4] # decoupled transfer matrices
        Dx = _np.array([[m[0,4]],[m[1,4]]])
        Dy = _np.array([[m[2,4]],[m[3,4]]])
        n = Twiss()
        n.spos   = t.spos + accelerator[i-1].length
        fp = closed_orbit[:,i]
        n.rx, n.px = fp[0], fp[1]
        n.ry, n.py = fp[2], fp[3]
        n.de, n.dl = fp[4], fp[5]
        n.betax  =  ((mx[0,0] * t.betax - mx[0,1] * t.alphax)**2 + mx[0,1]**2) / t.betax
        n.alphax = -((mx[0,0] * t.betax - mx[0,1] * t.alphax) * (mx[1,0] * t.betax - mx[1,1] * t.alphax) + mx[0,1] * mx[1,1]) / t.betax
        n.betay  =  ((my[0,0] * t.betay - my[0,1] * t.alphay)**2 + my[0,1]**2) / t.betay
        n.alphay = -((my[0,0] * t.betay - my[0,1] * t.alphay) * (my[1,0] * t.betay - my[1,1] * t.alphay) + my[0,1] * my[1,1]) / t.betay
        ''' calcs phase advance based on R(mu) = U(2) M(2|1) U^-1(1) '''
        n.mux = t.mux + _math.asin(mx[0,1]/_math.sqrt(n.betax * t.betax))
        n.muy = t.muy + _math.asin(my[0,1]/_math.sqrt(n.betay * t.betay))
        ''' dispersion function'''
        n.etax = Dx + _np.dot(mx, t.etax)
        n.etay = Dy + _np.dot(my, t.etay)

        tw.append(n)

        t = n.make_copy()

    ''' converts eta format '''
    for t in tw:
        t.etapx, t.etapy = (t.etax[1,0], t.etay[1,0])
        t.etax,  t.etay  = (t.etax[0,0], t.etay[0,0])

    return tw, m66, transfer_matrices, closed_orbit

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
def get_traces(accelerator, m66=None, closed_orbit=None):
    """Return traces of 6D one-turn transfer matrix"""
    if m66 is None:
        m66 = _tracking.findm66(accelerator,
                                indices = 'm66', closed_orbit = closed_orbit)
    trace_x = m66[0,0] + m66[1,1]
    trace_y = m66[2,2] + m66[3,3]
    trace_z = m66[4,4] + m66[5,5]
    return trace_x, trace_y, trace_z, m66, closed_orbit

@_interactive
def get_frac_tunes(accelerator, m66=None, closed_orbit=None):
    """Return fractional tunes of the accelerator"""
    trace_x, trace_y, trace_z, m66, closed_orbit = get_traces(accelerator,
                                                   m66 = m66,
                                                   closed_orbit = closed_orbit)
    tune_x = _math.acos(trace_x/2.0)/2.0/_math.pi
    tune_y = _math.acos(trace_y/2.0)/2.0/_math.pi
    tune_z = _math.acos(trace_z/2.0)/2.0/_math.pi
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
    _tracking.set4dtracking(accel)
    ring_length = _lattice.length(accel)

    dl = _np.zeros(_np.size(energy_offset))
    for i in range(len(energy_offset)):
        fp = _tracking.findorbit4(accel,energy_offset[i])
        X0 = _np.concatenate([fp,[energy_offset[i],0]]).tolist()
        T = _tracking.ringpass(accel,X0)
        dl[i] = T[0][5]/ring_length

    polynom = _np.polyfit(energy_offset,dl,order)
    a = _np.fliplr([polynom])[0].tolist()
    a = a[1:]
    if len(a) == 1:
        a=a[0]
    return a


@_interactive
def get_radiation_integrals2(accelerator,
                          twiss=None,
                          m66=None,
                          transfer_matrices=None,
                          closed_orbit=None):

    if twiss is None or m66 is None or transfer_matrices is None:
        fixed_point = closed_orbit if closed_orbit is None else closed_orbit[:,0]
        twiss, m66, transfer_matrices, closed_orbit = \
            calc_twiss(accelerator, fixed_point=fixed_point)

    t0 = _time.time()
    D_x, D_x_ = get_twiss(twiss,('etax','etapx'))
    gamma = _np.zeros(len(accelerator))
    integrals=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(accelerator)-1):
        gamma[i] = (1 + twiss[i].alphax**2) / twiss[i].betax
        gamma[i+1] = (1 + twiss[i+1].alphax**2) / twiss[i+1].betax
        if accelerator[i].angle != 0.0:
            rho = accelerator[i].length/accelerator[i].angle
            dispersion = 0.5*(D_x[i]+D_x[i+1])
            integrals[0] = integrals[0] + dispersion*accelerator[i].length /rho
            integrals[1] = integrals[1] + accelerator[i].length/(rho**2)
            integrals[2] = integrals[2] + accelerator[i].length/abs(rho**3)
            integrals[3] = integrals[3] + \
                D_x[i]*_math.tan(accelerator[i].angle_in)/(rho**2) + \
                (1 + 2*(rho**2)*accelerator[i].polynom_b[1])*(D_x[i]+D_x[i+1])*accelerator[i].length/(2*(rho**3)) + \
                D_x[i+1]*_math.tan(accelerator[i].angle_out)/(rho**2)
            H1 = twiss[i].betax*D_x_[i]*D_x_[i] + 2*twiss[i].alphax*D_x[i]*D_x_[i] + gamma[i]*D_x[i]*D_x[i];
            H0 = twiss[i+1].betax*D_x_[i+1]*D_x_[i+1] + 2*twiss[i+1].alphax*D_x[i+1]*D_x_[i+1] + gamma[i+1]*D_x[i+1]*D_x[i+1]
            integrals[4] = integrals[4] + accelerator[i].length*(H1+H0)*0.5/abs(rho**3)
            integrals[5] = integrals[5] + (accelerator[i].polynom_b[1]**2)*(dispersion**2)*accelerator[i].length
    t1 = _time.time()
    print(t1-t0)
    return integrals, twiss, m66, transfer_matrices, closed_orbit

@_interactive
def get_radiation_integrals(accelerator,
                          twiss=None,
                          m66=None,
                          transfer_matrices=None,
                          closed_orbit=None):
    """Calculate radiation integrals for periodic systems"""

    if twiss is None or m66 is None or transfer_matrices is None:
        fixed_point = closed_orbit if closed_orbit is None else closed_orbit[:,0]
        twiss, m66, transfer_matrices, closed_orbit = \
            calc_twiss(accelerator, fixed_point=fixed_point)

    #t0 = _time.time()
    spos,etax,etapx,betax,alphax = get_twiss(twiss,('spos','etax','etapx','betax','alphax'))
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

    #t1 = _time.time(); print(t1-t0)
    return integrals, twiss, m66, transfer_matrices, closed_orbit


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
                             transfer_matrices=None,
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

    integrals, *args = get_radiation_integrals(accelerator,twiss,m66,transfer_matrices,closed_orbit)

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
def get_beam_size(accelerator, coupling=0.0, closed_orbit=None):
    """Return beamsizes (stddev) along ring"""

    # twiss parameters
    twiss, *_ = calc_twiss(accelerator,closed_orbit=closed_orbit)
    betax, alphax, etax, etapx = get_twiss(twiss, ('betax','alphax','etax','etapx'))
    betay, alphay, etay, etapy = get_twiss(twiss, ('betay','alphay','etay','etapy'))
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
    return sigmax, sigmay, sigmaxl, sigmayl, ex, ey, summary, twiss, closed_orbit

@_interactive
def get_transverse_acceptance(accelerator, twiss=None, init_twiss=None, fixed_point=None, energy_offset=0.0):

    m66 = None
    if twiss is None:
        twiss, m66, transfer_matrices, closed_orbit = calc_twiss(accelerator, init_twiss=init_twiss, fixed_point=fixed_point)
    else:
        closed_orbit = _np.zeros((6,len(accelerator)))
        closed_orbit[0,:], closed_orbit[2,:] = get_twiss(twiss, ('rx','ry'))
    betax, betay, etax, etay = get_twiss(twiss, ('betax', 'betay', 'etax', 'etay'))
    # physical apertures
    lattice = accelerator._accelerator.lattice
    hmax, vmax = _np.array([(lattice[i].hmax,lattice[i].vmax) for i in range(len(accelerator))]).transpose()
    # calcs local linear acceptances
    co_x, co_y = closed_orbit[(0,2),:]

    n = len(accelerator)

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
        return accepx, accepy, twiss, closed_orbit
    else:
        return accepx, accepy, twiss, m66, transfer_matrices, closed_orbit
