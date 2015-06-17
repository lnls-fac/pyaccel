
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
        self.spos = None
        self.corx, self.copx  = 0, 0
        self.cory, self.copy  = 0, 0
        self.code, self.codl  = 0, 0
        self.etax, self.etaxl = 0, 0
        self.etay, self.etayl = 0, 0
        self.mux, self.betax, self.alphax = 0, None, None
        self.muy, self.betay, self.alphay = 0, None, None

    def __str__(self):
        r = ''
        r += 'spos      : ' + '{0:+10.3e}'.format(self.spos) + '\n'
        r += 'corx,copx : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.corx, self.copx) + '\n'
        r += 'cory,copy : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.cory, self.copy) + '\n'
        r += 'code,codl : ' + '{0:+10.3e}, {1:+10.3e}'.format(self.code, self.codl) + '\n'
        r += 'mux       : ' + '{0:+10.3e}'.format(self.mux) + '\n'
        r += 'betax     : ' + '{0:+10.3e}'.format(self.betax) + '\n'
        r += 'alphax    : ' + '{0:+10.3e}'.format(self.alphax) + '\n'
        r += 'etax      : ' + '{0:+10.3e}'.format(self.etax) + '\n'
        r += 'etaxl     : ' + '{0:+10.3e}'.format(self.etaxl) + '\n'
        r += 'muy       : ' + '{0:+10.3e}'.format(self.muy) + '\n'
        r += 'betay     : ' + '{0:+10.3e}'.format(self.betay) + '\n'
        r += 'alphay    : ' + '{0:+10.3e}'.format(self.alphay)
        return r

    def make_copy(self):
        n = Twiss()
        n.spos = self.spos
        n.corx, n.copx = self.corx, self.copx
        n.cory, n.copy = self.cory, self.copy
        n.code, n.codl = self.code, self.codl
        n.etax, n.etaxl = self.etax, self.etaxl
        n.etay, n.etayl = self.etay, self.etayl
        n.mux, n.betax, n.alphax = self.mux, self.betax, self.alphax
        n.muy, n.betay, n.alphay = self.muy, self.betay, self.alphay
        return n

    @staticmethod
    def make_new(spos=0.0, fixed_point=None, mu=None, beta=None, alpha=None, eta=None, etal=None):
        n = Twiss()
        if fixed_point is None:
            n.corx, n.copx, n.cory, n.copy, n.code, n.codl = (0.0,) * 6
        else:
            n.corx, n.copx, n.cory, n.copy, n.code, n.codl = fixed_point
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

@_interactive
def gettwiss(twiss_list, attribute_list):
    """Build a matrix with Twiss data from a list of Twiss objects.

    Accepts a list of Twiss objects and returns a matrix with Twiss data, one line for
    each Twiss parameter defined in 'attributes_list'.

    Keyword arguments:
    twiss_list -- List with Twiss objects
    attributes_list -- List of strings with Twiss attributes to be stored in twiss matrix

    Returns:
    m -- Matrix with Twiss data. Can also be thought of a single column of
         Twiss parameter vectors:
            betax, betay = gettwiss(twiss, ('betax','betay'))
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
def calctwiss(accelerator=None, init_twiss=None, fixed_point=None):
    """Return Twiss parameters of uncoupled dynamics."""

    if init_twiss is not None:
        ''' as a transport line: uses init_twiss '''
        if fixed_point is None:
            fixed_point = [init_twiss.corx, init_twiss.copx, init_twiss.cory, init_twiss.copy, init_twiss.code, init_twiss.codl]
        else:
            raise OpticsException('arguments init_twiss and fixed_orbit are mutually exclusive')
        closed_orbit, *_ = _tracking.linepass(accelerator, particles=list(fixed_point), indices = 'open')
        m66, transfer_matrices, *_ = _tracking.findm66(accelerator, closed_orbit = closed_orbit[0])
        mx, my = m66[0:2,0:2], m66[2:4,2:4]
        t = init_twiss
        t.etax = _np.array([[t.etax], [t.etaxl]])
        t.etay = _np.array([[t.etay], [t.etayl]])
    else:
        ''' as a periodic system: try to find periodic solution '''
        if fixed_point is None:
            if not accelerator.cavity_on and not accelerator.radiation_on:
                if accelerator.harmonic_number == 0:
                    raise OpticsException('Either harmonic number was not set or calctwiss was\
                    invoked for transport line without initial twiss')
                closed_orbit = _np.zeros((6,len(accelerator)))
                closed_orbit[:4,:] = _tracking.findorbit4(accelerator, indices='open')
            else:
                closed_orbit = _tracking.findorbit6(accelerator, indices='open')
        else:
            closed_orbit, *_ = _tracking.linepass(accelerator, particles=list(fixed_point), indices='open')

        ''' calcs twiss at first element '''
        m66, transfer_matrices, *_ = _tracking.findm66(accelerator, closed_orbit=closed_orbit)
        mx, my = m66[0:2,0:2], m66[2:4,2:4] # decoupled transfer matrices
        trace_x, trace_y, *_ = gettraces(accelerator, m66 = m66, closed_orbit=closed_orbit)
        if not (-2.0 < trace_x < 2.0):
            raise OpticsException('horizontal dynamics is unstable')
        if not (-2.0 < trace_y < 2.0):
            raise OpticsException('vertical dynamics is unstable')
        sin_nux = _math.copysign(1,mx[0,1]) * _math.sqrt(-mx[0,1] * mx[1,0] - ((mx[0,0] - mx[1,1])**2)/4);
        sin_nuy = _math.copysign(1,my[0,1]) * _math.sqrt(-my[0,1] * my[1,0] - ((my[0,0] - my[1,1])**2)/4);
        fp = closed_orbit[:,0]
        t = Twiss()
        t.spos = 0
        t.corx, t.copx = fp[0], fp[1]
        t.cory, t.copy = fp[2], fp[3]
        t.code, t.codl = fp[4], fp[5]
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
    for i in range(1, len(accelerator)):
        m = transfer_matrices[i-1]
        mx, my = m[0:2,0:2], m[2:4,2:4] # decoupled transfer matrices
        Dx = _np.array([[m[0,4]],[m[1,4]]])
        Dy = _np.array([[m[2,4]],[m[3,4]]])
        n = Twiss()
        n.spos   = t.spos + accelerator[i-1].length
        fp = closed_orbit[:,i]
        n.corx, n.copx = fp[0], fp[1]
        n.cory, n.copy = fp[2], fp[3]
        n.code, n.codl = fp[4], fp[5]
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
        t.etaxl, t.etayl = (t.etax[1,0], t.etay[1,0])
        t.etax,  t.etay  = (t.etax[0,0], t.etay[0,0])

    return tw, m66, transfer_matrices, closed_orbit

@_interactive
def getrffrequency(accelerator):
    """Return the frequency of the first RF cavity in the lattice"""
    for e in accelerator:
        if e.frequency != 0.0:
            return e.frequency
    else:
        raise OpticsException('no cavity element in the lattice')

@_interactive
def getrfvoltage(accelerator):
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
def getrevolutionperiod(accelerator):
    return accelerator.length/accelerator.velocity


@_interactive
def getrevolutionfrequency(accelerator):
    return 1.0/getrevolutionperiod(accelerator)


@_interactive
def gettraces(accelerator, m66 = None, closed_orbit = None):
    """Return traces of 6D one-turn transfer matrix"""
    if m66 is None:
        m66 = _tracking.findm66(accelerator,
                                indices = 'm66', closed_orbit = closed_orbit)
    trace_x = m66[0,0] + m66[1,1]
    trace_y = m66[2,2] + m66[3,3]
    trace_z = m66[4,4] + m66[5,5]
    return trace_x, trace_y, trace_z, m66, closed_orbit

@_interactive
def getfractunes(accelerator, m66 = None, closed_orbit = None):
    """Return fractional tunes of the accelerator"""
    trace_x, trace_y, trace_z, m66, closed_orbit = gettraces(accelerator,
                                                   m66 = m66,
                                                   closed_orbit = closed_orbit)
    tune_x = _math.acos(trace_x/2.0)/2.0/_math.pi
    tune_y = _math.acos(trace_y/2.0)/2.0/_math.pi
    tune_z = _math.acos(trace_z/2.0)/2.0/_math.pi
    return tune_x, tune_y, tune_z, trace_x, trace_y, trace_z, m66, closed_orbit


@_interactive
def getchromaticities(accelerator):
    raise OpticsException('not implemented')


@_interactive
def getmcf(accelerator, order=1, energy_offset=None):
    """Return momentum compaction factor of the accelerator"""
    if energy_offset is None:
        energy_offset = _np.linspace(-1e-3,1e-3,11)

    accel=accelerator[:]
    _tracking.set4dtracking(accel)
    ring_length = _lattice.lengthlat(accel)

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
def getradiationintegrals(accelerator=None,
                          twiss=None,
                          m66=None,
                          transfer_matrices=None,
                          closed_orbit=None):

    if twiss is None or m66 is None or transfer_matrices is None:
        fixed_point = closed_orbit if closed_orbit is None else closed_orbit[:,0]
        twiss, m66, transfer_matrices, closed_orbit = \
            calctwiss(accelerator, fixed_point=fixed_point)

    D_x, D_x_ = gettwiss(twiss,('etax','etaxl'))
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
    return integrals, twiss, m66, transfer_matrices, closed_orbit


@_interactive
def getnaturalenergyspread(accelerator):
    Cq = _mp.constants.Cq
    gamma = accelerator.gamma_factor
    integrals, *_ = getradiationintegrals(accelerator)
    natural_energy_spread = _math.sqrt( Cq*(gamma**2)*integrals[2]/(2*integrals[1] + integrals[3]))
    return natural_energy_spread


@_interactive
def getnaturalemittance(accelerator):
    Cq = _mp.constants.Cq
    gamma = accelerator.gamma_factor
    integrals, *_ = getradiationintegrals(accelerator)

    damping = _np.zeros(3)
    damping[0] = 1.0 - integrals[3]/integrals[1]
    damping[1] = 1.0
    damping[2] = 2.0 + integrals[3]/integrals[1]

    natural_emittance = Cq*(gamma**2)*integrals[4]/(damping[0]*integrals[1])
    return natural_emittance


@_interactive
def getnaturalbunchlength(accelerator):
    c = _mp.constants.light_speed
    rad_cgamma = _mp.constants.rad_cgamma
    e0 = accelerator.energy
    gamma = accelerator.gamma_factor
    beta = accelerator.beta_factor
    harmon = accelerator.harmonic_number

    integrals, *_ = getradiationintegrals(accelerator)
    rev_freq = getrevolutionfrequency(accelerator)
    compaction_factor = getmcf(accelerator)

    etac = gamma**(-2) - compaction_factor

    freq = getrffrequency(accelerator)
    v_cav = getrfvoltage(accelerator)
    radiation = rad_cgamma*((e0/1e9)**4)*integrals[1]/(2*_math.pi)*1e9
    overvoltage = v_cav/radiation

    syncphase = _math.pi - _math.asin(1/overvoltage)
    synctune = _math.sqrt((etac * harmon * v_cav * _math.cos(syncphase))/(2*_math.pi*e0))
    natural_energy_spread = getnaturalenergyspread(accelerator)
    bunchlength = beta* c *abs(etac)* natural_energy_spread /( synctune * rev_freq *2*_math.pi)
    return bunchlength

@_interactive
def getequilibriumparameters(accelerator,
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
    rev_freq = getrevolutionfrequency(accelerator)

    compaction_factor = getmcf(accelerator)
    etac = gamma**(-2) - compaction_factor

    integrals, *_ = getradiationintegrals(accelerator,twiss,m66,transfer_matrices,closed_orbit)

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

    freq = getrffrequency(accelerator)
    v_cav = getrfvoltage(accelerator)
    overvoltage = v_cav/radiation

    syncphase = _math.pi - _math.asin(1.0/overvoltage)
    synctune = _math.sqrt((etac * harmon * v_cav * _math.cos(syncphase))/(2*_math.pi*e0))
    rf_energy_acceptance = _math.sqrt(v_cav*_math.sin(syncphase)*2*(_math.sqrt((overvoltage**2)-1.0)
                        - _math.acos(1.0/overvoltage))/(_math.pi*harmon*abs(etac)*e0))
    bunchlength = beta* c *abs(etac)* natural_energy_spread /( synctune * rev_freq *2*_math.pi)

    summary=dict(compaction_factor = compaction_factor, radiation_integrals = integrals, damping_numbers = damping,
        damping_times = radiation_damping, natural_energy_spread = natural_energy_spread, etac = etac,
        natural_emittance = natural_emittance, overvoltage = overvoltage, syncphase = syncphase,
        synctune = synctune, rf_energy_acceptance = rf_energy_acceptance, bunchlength = bunchlength)
    return summary

@_interactive
def getbeamsize(accelerator, coupling=0.0, closed_orbit=None):
    """Return beamsizes (stddev) along ring"""

    # twiss parameters
    twiss, *_ = calctwiss(accelerator,closed_orbit=closed_orbit)
    betax, alphax, etax, etaxl = gettwiss(twiss, ('betax','alphax','etax','etaxl'))
    betay, alphay, etay, etayl = gettwiss(twiss, ('betay','alphay','etay','etayl'))
    gammax = (1.0 + alphax**2)/betax
    gammay = (1.0 + alphay**2)/betay
    # emittances and energy spread
    summary = getequilibriumparameters(accelerator)
    e0 = summary['natural_emittance']
    sigmae = summary['natural_energy_spread']
    ey = e0 * coupling / (1.0 + coupling)
    ex = e0 * 1 / (1.0 + coupling)
    # beamsizes per se
    sigmax  = _np.sqrt(ex * betax + (sigmae * etax)**2)
    sigmay  = _np.sqrt(ey * betay + (sigmae * etay)**2)
    sigmaxl = _np.sqrt(ex * gammax + (sigmae * etaxl)**2)
    sigmayl = _np.sqrt(ey * gammay + (sigmae * etayl)**2)
    return sigmax, sigmay, sigmaxl, sigmayl, ex, ey, summary, twiss, closed_orbit

@_interactive
def gettransverseacceptance(accelerator, twiss=None, init_twiss=None, fixed_point=None, energy_offset=0.0):

    m66 = None
    if twiss is None:
        twiss, m66, transfer_matrices, closed_orbit = calctwiss(accelerator, init_twiss=init_twiss, fixed_point=fixed_point)
    else:
        closed_orbit = _np.zeros((6,len(accelerator)))
        closed_orbit[0,:], closed_orbit[2,:] = gettwiss(twiss, ('corx','cory'))
    betax, betay, etax, etay = gettwiss(twiss, ('betax', 'betay', 'etax', 'etay'))
    # physical apertures
    hmax = _np.array(_lattice.getattributelat(accelerator, 'hmax'))
    vmax = _np.array(_lattice.getattributelat(accelerator, 'vmax'))
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
