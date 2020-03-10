import pyaccel
import pymodels

import numpy as np
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
    qe = 1.60217662e-19; # electron charge [C]
    r0 = 2.8179403227e-15; # classic electron radius [m]
    c  = 2.99792458e8;  # speed of light [m/s]
    Kb = 1.38e-23
    T=300
    epsilon0 = 8.854187817e-12; #Vacuum permittivity [F/m]
    m_e = 511e3#eV
    gamma = E0/m_e
   

    #Twiss
    twiss ,*_= pyaccel.optics.calc_twiss(model)
    n        = len(twiss.betax)
    s_B      = twiss.spos
    Bx       = twiss.betax
    By       = twiss.betay
    Ib = I/n_bunches
    T_rev    = pyaccel.optics.get_revolution_period(model)#s
    sig      = pyaccel.optics.get_beam_size(model)
    eta_x    = twiss.etax
    
    tau_es = elastic(Vx, Vy, Bx, By)

    tau_is = inelastic(accep, P)

    tau_T = Touschek (emit, acoplamento, Bx, By, eta_x, spread, sig_S, Ib)

    tau = ((1/tau_es+1/tau_is+1/tau_T)**-1)/3600

    return tau_es, tau_is, tau_T, tau 

def elastic(Vx, Vy, Bx, By, c, qe, epsilon0, Kb, Z, E0, T, P):
    
    #calcular aceitancias
    if (Vx<12e6):
        EA_x = 1e6*Vx**2 / Bx
    else:
        EA_x = 12**2 / Bx
    if (Vy<3e-3):
        EA_y = 1e6*Vy**2 / By
    else:
        EA_y = 3**2 / By
    

    #Calcula R
    theta_x = np.zeros(len(Bx))
    theta_y = np.zeros(len(Bx))
    Bmx = 518/49
    Bmy = 518/14
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
    print(2*math.pi/F[5])
    inv_es = Const*Z**2/(E0**2)/T/theta_y**2*F*P

    return inv_es**-1
    
def inelastic(accep, P):

    #Calcula tau inelastic
    Conv = 7.500616828e8
    inv_is = Conv*0.00653*math.log(1/abs(accep))*P/3600

    return inv_is**-1

def Touschek(emit, acoplamento, Bx, By, eta_x, spread, sig_S, Ib, T_rev, qe, r0, c, gamma):

    #Calcula Touschek
    Dp = 1e-1
    emitx = emit*(1-acoplamento)
    emity = emit*acoplamento
    V = np.zeros(len(Bx))
    for i in range (len(Bx)):
        sigx = math.sqrt(emitx*Bx[i]+(eta_x[i]*spread)**2)
        sigy = math.sqrt(emity*By[i])
        V[i] = sig_S * sigx * sigy
    N = Ib * T_rev / qe 
    inv_T = (r0**2*c/8/math.pi)*N/gamma**2/ accep**3 * Dp / V

    return (inv_T)**-1



tau = lifetime(model, abertura_x, abertura_y, energy_spread, vac, I, n_bunches, natural_emit, bunch_lengh, coupling, energy_accep)

print('Elastic lifetime is %f hours \n' %tau[0])

print('Inelastic lifetime is %f hours \n' %tau[1])

print('Touschek lifetime is %f hours \n' %tau[2])

print('Total lifetime is %f hours \n' %tau[3])
