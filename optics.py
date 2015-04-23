
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
        self.closed_orbit = _np.zeros((6,1))
        self.alphax = None
        self.betax  = None
        self.mux    = 0
        self.etaxl  = 0
        self.etax   = 0
        self.alphay = None
        self.betay  = None
        self.muy    = 0
        self.etayl  = 0
        self.etay   = 0
    def __str__(self):
        r = ''
        r += 'spos        : ' + '{0:+10.3e}'.format(self.spos) + '\n'
        r += 'closed orbit: ' + '{0[0][0]:+10.3e} {0[1][0]:+10.3e} {0[2][0]:+10.3e} {0[3][0]:+10.3e} {0[4][0]:+10.3e} {0[5][0]:+10.3e}'.format(self.closed_orbit) + '\n'
        r += 'mux         : ' + '{0:+10.3e}'.format(self.mux) + '\n'
        r += 'betax       : ' + '{0:+10.3e}'.format(self.betax) + '\n'
        r += 'alphax      : ' + '{0:+10.3e}'.format(self.alphax) + '\n'
        r += 'etax        : ' + '{0:+10.3e}'.format(self.etax) + '\n'
        r += 'etaxl       : ' + '{0:+10.3e}'.format(self.etaxl) + '\n'
        r += 'muy         : ' + '{0:+10.3e}'.format(self.muy) + '\n'
        r += 'betay       : ' + '{0:+10.3e}'.format(self.betay) + '\n'
        r += 'alphay      : ' + '{0:+10.3e}'.format(self.alphay) + '\n'
        r += 'etay        : ' + '{0:+10.3e}'.format(self.etay) + '\n'
        r += 'etayl       : ' + '{0:+10.3e}'.format(self.etayl) + '\n'
        return r


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
    values = _np.zeros((len(attribute_list),len(twiss_list)))
    for i in range(len(twiss_list)):
        for j in range(len(attribute_list)):
            values[j,i] = getattr(twiss_list[i], attribute_list[j])
    return values


@_interactive
def calctwiss(
        accelerator=None,
        indices=None,
        closed_orbit=None,
        twiss_in=None,
        all_data=False):
    """Return Twiss parameters of uncoupled dynamics."""

    ''' process arguments '''
    if indices is None:
        indices = range(len(accelerator))

    try:
        indices[0]
    except:
        indices = [indices]

    m66, transfer_matrices = _tracking.findm66(accelerator = accelerator, closed_orbit = closed_orbit)
    if twiss_in is None:
        twiss_in = Twiss()
        twiss_in.closed_orbit = closed_orbit
        twiss_in.mux, twiss_in.muy = 0, 0

    ''' calcs twiss at first element '''
    mx, my = m66[0:2,0:2], m66[2:4,2:4] # decoupled transfer matrices
    sin_nux = _math.copysign(1,mx[0,1]) * _math.sqrt(-mx[0,1] * mx[1,0] - ((mx[0,0] - mx[1,1])**2)/4);
    sin_nuy = _math.copysign(1,my[0,1]) * _math.sqrt(-my[0,1] * my[1,0] - ((my[0,0] - my[1,1])**2)/4);
    t = Twiss()
    t.spos    = 0
    t.alphax  = (mx[0,0] - mx[1,1]) / 2 / sin_nux
    t.betax   = mx[0,1] / sin_nux
    t.alphay  = (my[0,0] - my[1,1]) / 2 / sin_nuy
    t.betay   = my[0,1] / sin_nuy
    ''' dispersion function based on eta = (1 - M)^(-1) D'''
    Dx = _np.array([[m66[0,4]],[m66[1,4]]])
    Dy = _np.array([[m66[2,4]],[m66[3,4]]])
    t.etax = _np.linalg.solve(_np.eye(2,2) - mx, Dx)
    t.etay = _np.linalg.solve(_np.eye(2,2) - my, Dy)

    if 0 in indices:
        tw = [t]
    else:
        tw = []

    ''' propagates twiss through line '''
    m_previous = _np.eye(6,6)
    for i in range(1, len(accelerator)):
        m = transfer_matrices[i-1]
        mx, my = m[0:2,0:2], m[2:4,2:4] # decoupled transfer matrices
        Dx = _np.array([[m[0,4]],[m[1,4]]])
        Dy = _np.array([[m[2,4]],[m[3,4]]])
        n = Twiss()
        n.spos   = t.spos + accelerator[i-1].length
        n.betax  =  ((mx[0,0] * t.betax - mx[0,1] * t.alphax)**2 + mx[0,1]**2) / t.betax
        n.alphax = -((mx[0,0] * t.betax - mx[0,1] * t.alphax) * (mx[1,0] * t.betax - mx[1,1] * t.alphax) + mx[0,1] * mx[1,1]) / t.betax
        n.betay  =  ((my[0,0] * t.betay - my[0,1] * t.alphay)**2 + my[0,1]**2) / t.betay
        n.alphay = -((my[0,0] * t.betay - my[0,1] * t.alphay) * (my[1,0] * t.betay - my[1,1] * t.alphay) + my[0,1] * my[1,1]) / t.betay
        ''' calcs phase advance based on R(mu) = U(2) M(2|1) U^-1(1) '''
        n.mux    = t.mux + _math.asin(mx[0,1]/_math.sqrt(n.betax * t.betax))
        n.muy    = t.muy + _math.asin(my[0,1]/_math.sqrt(n.betay * t.betay))
        ''' dispersion function '''
        n.etax = Dx + _np.dot(mx, t.etax)
        n.etay = Dy + _np.dot(my, t.etay)

        if i in indices:
            tw.append(n)
        t = _copy.deepcopy(n)

    ''' converts eta format '''
    for t in tw:
        t.etaxl, t.etayl = (t.etax[1,0], t.etay[1,0])
        t.etax,  t.etay  = (t.etax[0,0], t.etay[0,0])

    if all_data:
        return tw, m66, transfer_matrices, closed_orbit
    else:
        return tw


@_interactive
def getrffrequency(accelerator):
    """Return the frequency of the first RF cavity in the lattice"""
    for e in accelerator:
        if e.frequency != 0:
            return e.frequency
    else:
        raise OpticsException('no cavity element in the lattice')

@_interactive
def getrfvoltage(accelerator):
    """Return the voltage of the first RF cavity in the lattice"""
    for e in accelerator:
        if e.voltage != 0:
            return e.voltage
    else:
        raise OpticsException('no cavity element in the lattice')

@_interactive
def getrevolutionperiod(accelerator):
    return accelerator.length/accelerator.velocity


@_interactive
def getrevolutionfrequency(accelerator):
    return 1.0/getrevolutionperiod(accelerator)


@_interactive
def getfractunes(accelerator, closed_orbit = None):
    m66 = _tracking.findm66(accelerator, indices = 'm66')
    trace_x = m66[0,0] + m66[1,1]
    trace_y = m66[2,2] + m66[3,3]
    trace_s = m66[4,4] + m66[5,5]
    tune_x = _math.acos(trace_x/2.0)/2.0/_math.pi
    tune_y = _math.acos(trace_y/2.0)/2.0/_math.pi
    tune_s = _math.acos(trace_s/2.0)/2.0/_math.pi
    return tune_x, tune_y, tune_s

@_interactive
def gettunes(accelerator):
    raise OpticsException('not implemented')


@_interactive
def getchromaticities(lattice):
    raise OpticsException('not implemented')


@_interactive
def getmcf(lattice):
    raise OpticsException('not implemented')


@_interactive
def getradiationintegrals(accelerator):
    tw = calctwiss(accelerator)
    D_x, D_x_ = gettwiss(tw,('etax','etaxl'))
    gamma = _np.zeros(len(accelerator))
    integrals=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(accelerator)-1):
        gamma[i] = (1 + tw[i].alphax**2) / tw[i].betax
        gamma[i+1] = (1 + tw[i+1].alphax**2) / tw[i+1].betax
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
            H1 = tw[i].betax*D_x_[i]*D_x_[i] + 2*tw[i].alphax*D_x[i]*D_x_[i] + gamma[i]*D_x[i]*D_x[i];
            H0 = tw[i+1].betax*D_x_[i+1]*D_x_[i+1] + 2*tw[i+1].alphax*D_x[i+1]*D_x_[i+1] + gamma[i+1]*D_x[i+1]*D_x[i+1]
            integrals[4] = integrals[4] + accelerator[i].length*(H1+H0)*0.5/abs(rho**3)
            integrals[5] = integrals[5] + (accelerator[i].polynom_b[1]**2)*(dispersion**2)*accelerator[i].length
    return integrals


@_interactive
def getbunchlength(accelerator):
    c = _mp.constants.light_speed
    E0 = _mp.constants.electron_rest_energy *_mp.constants._joule_2_eV
    rad_cgamma = _mp.constants.rad_cgamma
    Cq = _mp.constants.Cq

    e0 = accelerator.energy
    circumference = _lattice.lengthlat(accelerator)
    revFreq = getrevolutionfrequency(accelerator)
    gamma = e0/E0
    beta = _math.sqrt(1 - 1/gamma)

    compactionFactor = getmcf(accelerator) #not implemented
    etac = gamma**(-2) - compactionFactor

    naturalEnergySpread = _math.sqrt( Cq*(gamma**2)*integrals[2]/(2*integrals[1] + integrals[3]))

    freq = getrffrequency(accelerator)
    v_cav = getrfvoltage(accelerator)
    harmon = accelerator.harmonic_number

    radiation = rad_cgamma*((e0/1e9)**^4)*integrals[1]/(2*_math.pi)*1e9
    overvoltage = v_cav/radiation

    syncphase = _math.pi - _math.asin(1/overvoltage)
    synctune = _math.sqrt((etac * harmon * v_cav * _math.cos(syncphase))/(2*_math.pi*e0))

    bunchlength = beta* c *abs(etac)* naturalEnergySpread /( synctune * revFreq *2*_math.pi)
    return bunchlenght
