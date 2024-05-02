"""Equilibrium Parameters."""

import math as _math
import copy as _copy

import mathphys as _mp


class _EqParams:
    """Equilibrium parameters base class."""

    PARAMETERS = {
        'espread0',
        'bunlen',
        'U0', 'overvoltage', 'syncphase', 'synctune',
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

    @staticmethod
    def calc_U0(energy, energy_offset, I2):
        """."""
        E0 = energy / 1e9  # [GeV]
        E0 *= (1 + energy_offset)
        rad_cgamma = _mp.constants.rad_cgamma
        return rad_cgamma/(2*_math.pi) * E0**4 * I2 * 1e9  # [eV]

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
        fmtr = '{:33s}: '
        fmtn = '{:.4g}'
        fmte = fmtr + fmtn

        ints = 'Jx,Jy,Je'.split(',')
        rst += '\n' + fmti.format(', '.join(ints))
        rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])

        ints = 'taux,tauy,taue'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [ms]')
        rst += ', '.join([fmtn.format(1000*getattr(eqparam, x)) for x in ints])

        ints = 'alphax,alphay,alphae'.split(',')
        rst += '\n' + fmti.format(', '.join(ints) + ' [Hz]')
        rst += ', '.join([fmtn.format(getattr(eqparam, x)) for x in ints])

        rst += fmte.format('\nmomentum compaction x 1e4', eqparam.alpha*1e4)
        rst += fmte.format('\nenergy loss [keV]', eqparam.U0/1000)
        rst += fmte.format('\novervoltage', eqparam.overvoltage)
        rst += fmte.format('\nsync phase [°]', eqparam.syncphase*180/_math.pi)
        rst += fmte.format('\nsync tune', eqparam.synctune)
        rst += fmte.format('\nhorizontal emittance [nm.rad]', eqparam.emitx*1e9)
        rst += fmte.format('\nvertical emittance [pm.rad]', eqparam.emity*1e12)
        rst += fmte.format('\nnatural emittance [nm.rad]', eqparam.emit0*1e9)
        rst += fmte.format('\nnatural espread [%]', eqparam.espread0*100)
        rst += fmte.format('\nbunch length [mm]', eqparam.bunlen*1000)
        rst += fmte.format('\nRF energy accep. [%]', eqparam.rf_acceptance*100)
        return rst


class EqParamsNormalModes(_EqParams):
    """Equilibrium Parameters for normal modes."""

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
        fmtr = '{:33s}: '
        fmtn = '{:.4g}'

        fmte = fmtr + fmtn

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

        rst += fmte.format('\nmomentum compaction x 1e4', eqparam.alpha*1e4)
        rst += fmte.format('\nenergy loss [keV]', eqparam.U0/1000)
        rst += fmte.format('\novervoltage', eqparam.overvoltage)
        rst += fmte.format('\nsync phase [°]', eqparam.syncphase*180/_math.pi)
        rst += fmte.format('\nmode 1 emittance [nm.rad]', eqparam.emit1*1e9)
        rst += fmte.format('\nmode 2 emittance [pm.rad]', eqparam.emit2*1e12)
        rst += fmte.format('\nnatural espread [%]', eqparam.espread0*100)
        rst += fmte.format('\nbunch length [mm]', eqparam.bunlen*1000)
        rst += fmte.format('\nRF energy accep. [%]', eqparam.rf_acceptance*100)
        return rst