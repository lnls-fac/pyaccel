"""Equilibrium Parameters."""

import math as _math
import copy as _copy

import mathphys as _mp

from .miscellaneous import get_rf_voltage as _get_rf_voltage


class _EqParams:
    """Equilibrium parameters base class."""

    PARAMETERS = {
        'energy', 'energy_offset',
        'espread0', 'bunlen',
        'rf_voltage', 'U0', 'overvoltage', 'syncphase', 'synctune',
        'alpha', 'etac', 'rf_acceptance',
        'sigma_rx', 'sigma_px',
        'sigma_ry', 'sigma_py',
    }

    def __init__(self, eqparams=None):
        """."""
        # initialize parameters with corresponding eqparam values or with None.
        for param in self.PARAMETERS:
            if isinstance(eqparams, dict):
                value = eqparams.get(param, None)
            else:
                value = getattr(eqparams, param, None)
            value = _copy.deepcopy(value)
            setattr(self, param, value)

    def __str__(self):
        """."""
        return self.eqparam_to_string(self)


class EqParamsXYModes(_EqParams):
    """Equilibrium parameters for XY modes."""

    PARAMETERS = _EqParams.PARAMETERS.union({
        'Jx', 'Jy', 'Je',
        'alphax', 'alphay', 'alphae',
        'taux', 'tauy', 'taue',
        'emitx', 'emity', 'emit0',
    })

    @staticmethod
    def eqparam_to_string(eqparam):
        """."""
        rst = ''
        fmti = '{:32s}: '
        fmtn = '{:.4g}'
        fmte = fmti + fmtn
        prechar = ''

        if eqparam.energy is not None:
            rst += fmte.format(prechar + 'Energy [GeV]', eqparam.energy*1e-9)
            prechar = '\n'
        if eqparam.energy_offset is not None:
            rst += prechar + fmte.format(
                'Energy offset [%]', eqparam.energy_offset*100)
            prechar = '\n'

        if None not in (eqparam.Jx, eqparam.Jy, eqparam.Je):
            ints = 'Jx,Jy,Je'.split(',')
            rst += prechar + fmti.format(', '.join(ints))
            rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])
            prechar = '\n'

        if None not in (eqparam.taux, eqparam.tauy, eqparam.taue):
            ints = 'taux,tauy,taue'.split(',')
            rst += prechar + fmti.format(', '.join(ints) + ' [ms]')
            rst += ', '.join([fmtn.format(
                1000*getattr(eqparam, x)) for x in ints])
            prechar = '\n'

        if None not in (eqparam.alphax, eqparam.alphay, eqparam.alphae):
            ints = 'alphax,alphay,alphae'.split(',')
            rst += prechar + fmti.format(', '.join(ints) + ' [Hz]')
            rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])
            prechar = '\n'

        if eqparam.alpha is not None:
            rst += prechar + fmte.format(
                'momentum compaction x 1e4', eqparam.alpha*1e4)
            prechar = '\n'
        if eqparam.U0 is not None:
            rst += prechar + fmte.format('energy loss [keV]', eqparam.U0/1000)
            prechar = '\n'
        if eqparam.overvoltage is not None:
            rst += prechar + fmte.format(
                'overvoltage', eqparam.overvoltage)
            prechar = '\n'
        if eqparam.syncphase is not None:
            rst += prechar + fmte.format(
                'sync phase [°]', eqparam.syncphase*180/_math.pi)
            prechar = '\n'
        if eqparam.synctune is not None:
            rst += prechar + fmte.format('sync tune', eqparam.synctune)
            prechar = '\n'
        if eqparam.emitx is not None:
            rst += prechar + fmte.format(
                'horizontal emittance [pm.rad]', eqparam.emitx*1e12)
            prechar = '\n'
        if eqparam.emity is not None:
            rst += prechar + fmte.format(
                'vertical emittance [pm.rad]', eqparam.emity*1e12)
            prechar = '\n'
        if eqparam.emit0 is not None:
            rst += prechar + fmte.format(
                'natural emittance [pm.rad]', eqparam.emit0*1e12)
            prechar = '\n'
        if eqparam.espread0 is not None:
            rst += prechar + fmte.format(
                'natural espread [%]', eqparam.espread0*100)
            prechar = '\n'
        if eqparam.bunlen is not None:
            rst += prechar + fmte.format(
                'bunch length [mm]', eqparam.bunlen*1000)
            prechar = '\n'
        if eqparam.rf_acceptance is not None:
            rst += prechar + fmte.format(
                'RF energy accep. [%]', eqparam.rf_acceptance*100)
            prechar = '\n'
        return rst


class EqParamsNormalModes(_EqParams):
    """Equilibrium Parameters for normal modes."""

    CHR_MODE1, CHR_MODE2, CHR_MODE3 = '1', '2', '3'
    PARAMETERS = _EqParams.PARAMETERS.union({
        'J1', 'J2', 'J3',
        'alpha1', 'alpha2', 'alpha3',
        'tau1', 'tau2', 'tau3',
        'tune1', 'tune2', 'tune3',
        'emit1', 'emit2',
        'tilt_xyplane',
    })

    @staticmethod
    def eqparam_to_string(eqparam):
        """."""
        rst = ''
        fmti = '{:32s}: '
        fmtn = '{:.4g}'
        fmte = fmti + fmtn
        prechar = ''

        if eqparam.energy is not None:
            rst += fmte.format(prechar + 'Energy [GeV]', eqparam.energy*1e-9)
            prechar = '\n'
        if eqparam.energy_offset is not None:
            rst += prechar + fmte.format(
                'Energy offset [%]', eqparam.energy_offset*100)
            prechar = '\n'
        if None not in (eqparam.J1, eqparam.J2, eqparam.J3):
            ints = 'J1,J2,J3'.split(',')
            rst += prechar + fmti.format(', '.join(ints))
            rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])
            prechar = '\n'

        if None not in (eqparam.tau1, eqparam.tau2, eqparam.tau3):
            ints = 'tau1,tau2,tau3'.split(',')
            rst += prechar + fmti.format(', '.join(ints) + ' [ms]')
            rst += ', '.join([fmtn.format(
                1000*getattr(eqparam, x)) for x in ints])
            prechar = '\n'

        if None not in (eqparam.alpha1, eqparam.alpha2, eqparam.alpha3):
            ints = 'alpha1,alpha2,alpha3'.split(',')
            rst += prechar + fmti.format(', '.join(ints) + ' [Hz]')
            rst += ', '.join([fmtn.format(
                getattr(eqparam, x)) for x in ints])
            prechar = '\n'

        if None not in (eqparam.tune1, eqparam.tune2, eqparam.tune3):
            ints = 'tune1,tune2,tune3'.split(',')
            rst += prechar + fmti.format(', '.join(ints) + ' [Hz]')
            rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])
            prechar = '\n'

        if eqparam.alpha is not None:
            rst += prechar + fmte.format(
                'momentum compaction x 1e4', eqparam.alpha*1e4)
            prechar = '\n'
        if eqparam.U0 is not None:
            rst += prechar + fmte.format('energy loss [keV]', eqparam.U0/1000)
            prechar = '\n'
        if eqparam.overvoltage is not None:
            rst += prechar + fmte.format('overvoltage', eqparam.overvoltage)
            prechar = '\n'
        if eqparam.syncphase is not None:
            rst += prechar + fmte.format(
                'sync phase [°]', eqparam.syncphase*180/_math.pi)
            prechar = '\n'
        if eqparam.emit1 is not None:
            rst += prechar + fmte.format(
                'mode 1 emittance [nm.rad]', eqparam.emit1*1e9)
            prechar = '\n'
        if eqparam.emit2 is not None:
            rst += prechar + fmte.format(
                'mode 2 emittance [pm.rad]', eqparam.emit2*1e12)
            prechar = '\n'
        if eqparam.espread0 is not None:
            rst += prechar + fmte.format(
                'natural espread [%]', eqparam.espread0*100)
            prechar = '\n'
        if eqparam.bunlen is not None:
            rst += prechar + fmte.format(
                'bunch length [mm]', eqparam.bunlen*1000)
            prechar = '\n'
        if eqparam.rf_acceptance is not None:
            rst += prechar + fmte.format(
                'RF energy accep. [%]', eqparam.rf_acceptance*100)
            prechar = '\n'
        return rst
