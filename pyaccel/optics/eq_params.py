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
        # initialize parameters with corresponding eqparam values or with none.
        for param in self.PARAMETERS:
            value = getattr(eqparams, param, None)
            value = _copy.deepcopy(value)
            setattr(self, param, value)

    def __str__(cls):
        """."""
        return cls.eqparam_to_string(cls)


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

        rst += fmte.format('Energy [GeV]', eqparam.energy*1e-9)
        rst += '\n' + fmte.format('Energy offset [%]', eqparam.energy_offset*100)

        ints = 'Jx,Jy,Je'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])

        ints = 'taux,tauy,taue'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [ms]')
        rst += ', '.join([fmtn.format(1000*getattr(eqparam, x)) for x in ints])

        ints = 'alphax,alphay,alphae'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [Hz]')
        rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])

        rst += '\n' + fmte.format('momentum compaction x 1e4', eqparam.alpha*1e4)
        rst += '\n' + fmte.format('energy loss [keV]', eqparam.U0/1000)
        rst += '\n' + fmte.format('overvoltage', eqparam.overvoltage)
        rst += '\n' + fmte.format('sync phase [°]', eqparam.syncphase*180/_math.pi)
        rst += '\n' + fmte.format('sync tune', eqparam.synctune)
        rst += '\n' + fmte.format('horizontal emittance [nm.rad]', eqparam.emitx*1e9)
        rst += '\n' + fmte.format('vertical emittance [pm.rad]', eqparam.emity*1e12)
        rst += '\n' + fmte.format('natural emittance [nm.rad]', eqparam.emit0*1e9)
        rst += '\n' + fmte.format('natural espread [%]', eqparam.espread0*100)
        rst += '\n' + fmte.format('bunch length [mm]', eqparam.bunlen*1000)
        rst += '\n' + fmte.format('RF energy accep. [%]', eqparam.rf_acceptance*100)
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

        rst += fmte.format('Energy [GeV]', eqparam.energy*1e-9)
        rst += '\n' + fmte.format('Energy offset [%]', eqparam.energy_offset*100)

        ints = 'J1,J2,J3'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])

        ints = 'tau1,tau2,tau3'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [ms]')
        rst += ', '.join([fmtn.format(1000*getattr(eqparam, x)) for x in ints])

        ints = 'alpha1,alpha2,alpha3'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [Hz]')
        rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])

        ints = 'tune1,tune2,tune3'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [Hz]')
        rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])

        rst += '\n' + fmte.format('momentum compaction x 1e4', eqparam.alpha*1e4)
        rst += '\n' + fmte.format('energy loss [keV]', eqparam.U0/1000)
        rst += '\n' + fmte.format('overvoltage', eqparam.overvoltage)
        rst += '\n' + fmte.format('sync phase [°]', eqparam.syncphase*180/_math.pi)
        rst += '\n' + fmte.format('mode 1 emittance [nm.rad]', eqparam.emit1*1e9)
        rst += '\n' + fmte.format('mode 2 emittance [pm.rad]', eqparam.emit2*1e12)
        rst += '\n' + fmte.format('natural espread [%]', eqparam.espread0*100)
        rst += '\n' + fmte.format('bunch length [mm]', eqparam.bunlen*1000)
        rst += '\n' + fmte.format('RF energy accep. [%]', eqparam.rf_acceptance*100)
        return rst