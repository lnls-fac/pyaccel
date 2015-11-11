
import numpy as _np
import mathphys as _mp
import pyaccel.optics as _optics
import pyaccel.lattice as _lattice
from pyaccel.utils import interactive as _interactive


@_interactive
def calc_lifetimes(accelerator, n=None, coupling=None, pressure_profile=None, twiss=None, eq_parameters=None):

    parameters, twiss = _process_args(accelerator, twiss, eq_parameters, n, coupling, pressure_profile)

    # Acceptances
    energy_acceptance = parameters['rf_energy_acceptance']
    accepx, accepy, *_ = _optics.get_transverse_acceptance(accelerator, twiss, energy_offset=0.0)
    transverse_acceptances = [min(accepx), min(accepy)]

    # Average pressure
    s, pressure  = pressure_profile
    avg_pressure = _np.trapz(pressure,s)/(s[-1]-s[0])

    # Loss rates
    spos        = _lattice.find_spos(accelerator)
    e_rate_spos = _mp.beam_lifetime.calc_elastic_loss_rate(transverse_acceptances, avg_pressure, z=7, temperature=300, **parameters)
    e_rate      = _np.trapz(e_rate_spos,spos)/(spos[-1]-spos[0])
    i_rate      = _mp.beam_lifetime.calc_inelastic_loss_rate(energy_acceptance, avg_pressure, z=7, temperature=300)
    q_rate      = sum(_mp.beam_lifetime.calc_quantum_loss_rates(transverse_acceptances, energy_acceptance, coupling, **parameters))
    tous_lt     = _mp.beam_lifetime.calc_touschek_loss_rate([-energy_acceptance,energy_acceptance], twiss, coupling, n, **parameters)

    # Lifetimes
    e_lifetime = float("inf") if e_rate == 0.0 else 1.0/e_rate
    i_lifetime = float("inf") if i_rate == 0.0 else 1.0/i_rate
    q_lifetime = float("inf") if q_rate == 0.0 else 1.0/q_rate
    t_coeff    = tous_lt['ave_rate']

    return e_lifetime, i_lifetime, q_lifetime, t_coeff


def _process_args(accelerator, twiss=None, eq_parameters=None, n=None, coupling=None, pressure_profile=None):

    m66 = None ; closed_orbit = None

    if twiss is None:
        twiss, m66 = _optics.calc_twiss(accelerator)
    if eq_parameters is None:
        eq_parameters, *_ = _optics.get_equilibrium_parameters(accelerator, twiss, m66)
    if n is None:
        raise Exception('Number of electrons per bunch was not set')
    if coupling is None:
        raise Exception('Coupling coefficient was not set')
    if pressure_profile is None:
        raise Exception('Pressure profile was not set')

    betax, betay, alphax = twiss.betax, twiss.betay, twiss.alphax
    etax,  etay,  etapx  = twiss.etax,  twiss.etay,  twiss.etapx

    parameters = {}
    parameters.update(eq_parameters)
    parameters['energy'] = accelerator.energy
    parameters['energy_spread'] = parameters.pop('natural_energy_spread')
    parameters['betax'], parameters['betay']  = betax, betay
    parameters['etax'], parameters['etay']    = etax, etay
    parameters['alphax'], parameters['etapx'] = alphax, etapx

    return parameters, twiss
