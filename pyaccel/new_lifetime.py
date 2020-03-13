import pyaccel
import pymodels
import mathphys

import numpy as np
from scipy import integrate
import math

#Life time calculations

model = pymodels.si.create_accelerator()
abertura_x = 12e-3 #m
abertura_y = 2e-3 #m
energy_spread = 8.7e-4
vac = 1e-9 #mTorr
I = 100e-3 #A
n_bunches = 864
natural_emit = 0.25e-9 # m rad
bunch_lengh = 2.3e-3 #m
coupling = 0.01
energy_accep = 0.03
ring_loc = 0




def lifetime(model, Vx, Vy, spread, P, I, n_bunches, emit, sig_S, acoplamento, accep):
    lattice = pymodels.si.lattice.create_lattice()
    #parameters
    Z=7 #atomic number
    E0 = model.energy #energia do feixe (eV)
    mcf = pyaccel.optics.get_mcf(model) #compaction factor
    J = pyaccel.optics.EquilibriumParameters.Je #damping factors
    E_n = pyaccel.optics.EquilibriumParameters.emit0#naturalEmittance;
    tau_am = pyaccel.optics.EquilibriumParameters.taue #radiationDamping;
    k =  pymodels.si.harmonic_number   #harmon;
    q = pyaccel.optics.EquilibriumParameters.overvoltage
    qe = mathphys.constants.elementary_charge; # electron charge [C]
    r0 = mathphys.constants.electron_radius; # classic electron radius [m]
    c  = mathphys.constants.light_speed;  # speed of light [m/s]
    Kb = mathphys.constants.boltzmann_constant
    T=300
    epsilon0 = mathphys.constants.vacuum_permitticity #Vacuum permittivity [F/m]
    m_e = mathphys.constants.electron_rest_energy/qe#eV
    gamma = E0/m_e
   

    #Twiss
    twiss ,*_= pyaccel.optics.calc_twiss(model)
    n        = len(twiss.betax)
    s_B      = twiss.spos
    Bx       = twiss.betax
    By       = twiss.betay
    Ib = I/n_bunches
    T_rev    = pyaccel.optics.get_revolution_period(model)#s
    eta_x    = twiss.etax
    length   = pyaccel.lattice.get_attribute(model, 'length')
    L        = sum(length)
    Vac_x    = np.array(pyaccel.lattice.get_attribute(model, 'vmax'))-np.array(pyaccel.lattice.get_attribute(model, 'vmin'))
    Vac_y    = np.array(pyaccel.lattice.get_attribute(model, 'hmax'))-np.array(pyaccel.lattice.get_attribute(model, 'hmin'))
    
    tau_es = elastic(Vx, Vy, Bx, By, c, qe, epsilon0, Kb, Z, E0, T, P, length, L, Vac_x, Vac_y)/3600

    tau_is = inelastic(accep, P)/3600

    tau_T = Touschek (emit, acoplamento, Bx, By, eta_x, spread, accep, sig_S, Ib, T_rev, qe, r0, c, gamma)/3600

    tau = ((1/tau_es+1/tau_is+1/tau_T)**-1)

    return tau_es, tau_is, tau_T, tau 

def elastic(Vx, Vy, Bx, By, c, qe, epsilon0, Kb, Z, E0, T, P, length, L, Vac_x, Vac_y):
    #Calcular valores limite
    Ex = Vac_x**2/Bx
    Ey = Vac_y**2/By

    Vac_xmin =  np.amin(Ex)
    Vac_ymin =  np.amin(Ey)

    #calcular valor scrapper
    Scr_x = Vx**2 / Bx
    Scr_y = Vy**2 / By
    
    EA_x = np.zeros(len(Bx))
    EA_y = np.zeros(len(Bx))
    for i in range (len(Bx)):
        if Scr_x[i]<Vac_xmin:
            EA_x[i] = 1e6*Scr_x[i]
        else:
            EA_x[i] = 1e6*Vac_xmin
        if Scr_y[i]<Vac_ymin:
            EA_y[i] = 1e6*Scr_y[i]
        else:
            EA_y[i] = 1e6*Vac_ymin
    

    #Calcula R
    theta_x = np.zeros(len(Bx))
    theta_y = np.zeros(len(Bx))
    #Bmx = L/49
    #Bmy = L/14

    BxL = Bx*length
    ByL = Bx*length
    Bmx = sum(BxL)/L
    Bmy = sum(ByL)/L

    for i in range (len(Bx)):
        theta_x[i] = math.sqrt(EA_x[i] /Bmx)
        theta_y[i] = math.sqrt(EA_y[i] /Bmy)
    R = theta_y / theta_x
    
    #Calcula tau elastic scattering
    Const = c*qe**4/(4*math.pi**2*epsilon0**2*Kb); # [m, K, J, m*rad, Pa]
    Const = Const*1e8/(qe)**2; # [m, K, eV, mm*mrad, mbar]
    F = np.zeros(len(Bx))
    for i in range (len(Bx)):
        F[i] =  (math.pi+(R[i]**2+1)*math.sin(2*math.atan(R[i]))+2*(R[i]**2-1)*math.atan(R[i]))
    inv_es = Const*Z**2/(E0**2)/T/theta_y**2*F*P

    return inv_es[ring_loc]**-1
    
def inelastic(accep, P):

    #Calcula tau inelastic
    Conv = 7.500616828e8
    inv_is = Conv*0.00653*math.log(1/abs(accep))*P/3600

    return inv_is**-1

def Touschek(emit, acoplamento, Bx, By, eta_x, spread, accep, sig_S, Ib, T_rev, qe, r0, c, gamma):

    #Calcula Touschek
    
    emitx = emit*(1-acoplamento)
    emity = emit*acoplamento
    V = np.zeros(len(Bx))
    sigx = np.zeros(len(Bx))
    Dp = np.zeros(len(Bx))
    for i in range (len(Bx)):
        sigx[i] = math.sqrt(emitx*Bx[i]+(eta_x[i]*spread)**2)
        sigy = math.sqrt(emity*By[i])
        V[i] = sig_S * sigx[i] * sigy
        Dp[i] = Calc_D(accep, sigx[i], gamma)

    N = Ib * T_rev / qe 
    inv_T = (r0**2*c/8/math.pi)*N/gamma**2/ accep**3 * Dp / V

    return (inv_T[ring_loc])**-1

def Calc_D(accep, sigx,gamma):
    eps = (accep/(sigx*gamma))**2

    f1 = lambda x : np.exp(-x)*np.log(x)/x
    f2 = lambda x : np.exp(-x)/x
    I1 = integrate.fixed_quad(f1, eps, 100*eps, n=1000)
    I2 = integrate.fixed_quad(f2, eps, 100*eps, n=1000)
    D = np.sqrt(eps)*(-3*np.exp(-eps)/2+eps*I1[0]/2+(3*eps-eps*np.log(eps)+2)*I2[0]/2)
    return D

tau = lifetime(model, abertura_x, abertura_y, energy_spread, vac, I, n_bunches, natural_emit, bunch_lengh, coupling, energy_accep)

print('Elastic lifetime is %f hours \n' %tau[0])

print('Inelastic lifetime is %f hours \n' %tau[1])

print('Touschek lifetime is %f hours \n' %tau[2])

print('Total lifetime is %f hours \n' %tau[3])
