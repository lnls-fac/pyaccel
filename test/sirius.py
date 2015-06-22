import math as _math
import pyaccel as _pyaccel
import mathphys as _mp

_harmonic_number  = 864
_energy = 3.0e9 # [eV]
_default_cavity_on = False
_default_radiation_on = False
_default_vchamber_on = False

def create_accelerator():

    accelerator = _pyaccel.accelerator.Accelerator(
            lattice=create_lattice(),
            energy=_energy,
            harmonic_number=_harmonic_number,
            cavity_on=_default_cavity_on,
            radiation_on=_default_radiation_on,
            vchamber_on=_default_vchamber_on)

    return accelerator

def create_lattice():


    _strengths_C05 = {

        #  QUADRUPOLOS
        #  ===========

        'qfa' :  4.500939083585109,
        'qda' : -2.948507308101822,
        'qdb2': -2.607851320364556,
        'qfb' :  4.468536646332483,
        'qdb1': -3.602260332930565,
        'qf1' :  3.107153201659299,
        'qf2' :  4.091315394710002,
        'qf3' :  3.586569278001512,
        'qf4' :  3.709349513079713,

        #  SEXTUPOLOS
        #  ===========
        'sda' : -46.758679575198470,
        'sfa' :  26.6622074299220,
        'sdb' : -49.571445128975940,
        'sfb' :  66.18257437452495,
        'sd1' : -174.15370941085680,
        'sf1' :  197.37251360249565,
        'sd2' : -93.85881072190855,
        'sd3' : -125.64988560404710,
        'sf2' :  142.15121115328305,
        'sd6' : -133.2016966347285,
        'sf4' :  229.65804021532680,
        'sd5' : -123.19826994348845,
        'sd4' : -173.79053483404260,
        'sf3' :  215.66183452267105,

    }

    # -- shortcut symbols --
    marker = _pyaccel.elements.marker
    drift = _pyaccel.elements.drift
    quadrupole = _pyaccel.elements.quadrupole
    sextupole = _pyaccel.elements.sextupole
    rbend_sirius = _pyaccel.elements.rbend
    rfcavity = _pyaccel.elements.rfcavity
    strengths = _strengths_C05

    # -- drifts --
    LIA = drift('lia', 2.4129)
    LIB = drift('lib', 2.0429)
    LBC = drift('lbc', 0.0630)
    L11 = drift('l11', 0.1100)
    L12 = drift('l12', 0.1200)
    L13 = drift('l13', 0.1300)
    L14 = drift('l14', 0.1400)
    L15 = drift('l15', 0.1500)
    L17 = drift('l17', 0.1700)
    L19 = drift('l19', 0.1900)
    L21 = drift('l21', 0.2100)
    L23 = drift('l23', 0.2300)
    L24 = drift('l24', 0.2400)
    L25 = drift('l25', 0.2500)
    L34 = drift('l34', 0.3400)
    L38 = drift('l38', 0.3800)
    L46 = drift('l46', 0.4600)
    L48 = drift('l48', 0.4800)
    L50 = drift('l50', 0.5000)
    L61 = drift('l61', 0.6100)
    L73 = drift('l73', 0.7300)

    # -- lattice markers --
    START    = marker('start')          # start of the model
    END      = marker('end')            # end of the model
    MIA      = marker('mia')            # center of long straight sections (even-numbered)
    MIB      = marker('mib')            # center of short straight sections (odd-numbered)
    GIRDER   = marker('girder')         # marker used to delimitate girders. one marker at begin and another at end of girder.
    MIDA     = marker('id_enda')        # marker for the extremities of IDs in long straight sections
    MIDB     = marker('id_endb')        # marker for the extremities of IDs in short straight sections
    MOMACCEP = marker('calc_mom_accep') # marker to define points where momentum acceptance will be calculated


    # -- dipoles --
    deg2rad = _math.pi/180.0

    B1E = rbend_sirius('b1', 0.828/2,  2.7553*deg2rad/2, 1.4143*deg2rad/2, 0,   0, 0, 0, [0, 0, 0], [0, -0.78, 0])
    MB1 = marker('mb1')
    B1S = rbend_sirius('b1', 0.828/2,  2.7553*deg2rad/2, 0, 1.4143*deg2rad/2,   0, 0, 0, [0, 0, 0], [0, -0.78, 0])
    B1  = [MOMACCEP,B1E,MOMACCEP,MB1,B1S,MOMACCEP]

    B2E = rbend_sirius('b2', 1.231/3, 4.0964*deg2rad/3, 1.4143*deg2rad/2, 0,   0, 0, 0, [0, 0, 0], [0, -0.78, 0])
    B2M = rbend_sirius('b2', 1.231/3, 4.0964*deg2rad/3, 0, 0,   0, 0, 0, [0, 0, 0], [0, -0.78, 0])
    B2S = rbend_sirius('b2', 1.231/3, 4.0964*deg2rad/3, 0, 1.4143*deg2rad/2,   0, 0, 0, [0, 0, 0], [0, -0.78, 0])
    B2  = [MOMACCEP,B2E,MOMACCEP,B2M,MOMACCEP,B2S,MOMACCEP]

    B3E = rbend_sirius('b3', 0.425/2, 1.4143*deg2rad/2, 1.4143*deg2rad/2, 0,   0, 0, 0, [0, 0, 0], [0, -0.78, 0])
    MB3 = marker('mb3')
    B3S = rbend_sirius('b3', 0.425/2, 1.4143*deg2rad/2, 0, 1.4143*deg2rad/2,   0, 0, 0, [0, 0, 0], [0, -0.78, 0])
    B3  = [MOMACCEP,B3E,MOMACCEP,MB3,B3S,MOMACCEP]

    BC1 = rbend_sirius('bc', 0.01,  0.114*deg2rad, 0, 0,   0, 0, 0, [0, 0, 0], [0, -0.0001586, -28.62886])
    BC2 = rbend_sirius('bc', 0.01,  0.110*deg2rad, 0, 0,   0, 0, 0, [0, 0, 0], [0, -0.0032399, -27.61427])
    BC3 = rbend_sirius('bc', 0.01,  0.096*deg2rad, 0, 0,   0, 0, 0, [0, 0, 0], [0, -0.0126344, -21.90036])
    BC4 = rbend_sirius('bc', 0.01,  0.078*deg2rad, 0, 0,   0, 0, 0, [0, 0, 0], [0, -0.0163734, -14.03515])
    BC5 = rbend_sirius('bc', 0.01,  0.062*deg2rad, 0, 0,   0, 0, 0, [0, 0, 0], [0, -0.0145890, -8.797988])
    BC6 = rbend_sirius('bc', 0.08,  0.274*deg2rad, 0, 0,   0, 0, 0, [0, 0, 0], [0, -0.0055834, -4.294111])
    MC = marker('mc')
    BCE = [MOMACCEP, BC6, BC5, BC4, BC3, BC2, BC1]
    BCS = [BC1, BC2, BC3, BC4, BC5, BC6, MOMACCEP]
    BC  = [BCE,MC, MOMACCEP,BCS]

    # -- quadrupoles --
    QFA  = quadrupole('qfa',  0.200, strengths['qfa'])
    QDA  = quadrupole('qda',  0.140, strengths['qda'])
    QDB2 = quadrupole('qdb2', 0.140, strengths['qdb2'])
    QFB  = quadrupole('qfb',  0.300, strengths['qfb'])
    QDB1 = quadrupole('qdb1', 0.140, strengths['qdb1'])
    QF1  = quadrupole('qf1',  0.200, strengths['qf1'])
    QF2  = quadrupole('qf2',  0.200, strengths['qf2'])
    QF3  = quadrupole('qf3',  0.200, strengths['qf3'])
    QF4  = quadrupole('qf4',  0.200, strengths['qf4'])

    # -- sextupoles and slow correctors --
    SDA = sextupole('sda', 0.150, strengths['sda']) #
    SFA = sextupole('sfa', 0.150, strengths['sfa']) # chs/cvs
    SDB = sextupole('sdb', 0.150, strengths['sdb']) #
    SFB = sextupole('sfb', 0.150, strengths['sfb']) # chs/cvs
    SD1 = sextupole('sd1', 0.150, strengths['sd1']) # chs/cvs
    SF1 = sextupole('sf1', 0.150, strengths['sf1']) #
    SD2 = sextupole('sd2', 0.150, strengths['sd2']) # chs
    SD3 = sextupole('sd3', 0.150, strengths['sd3']) # cvs
    SF2 = sextupole('sf2', 0.150, strengths['sf2']) # chs
    SD6 = sextupole('sd6', 0.150, strengths['sd6']) # chs/cvs
    SF4 = sextupole('sf4', 0.150, strengths['sf4']) #
    SD5 = sextupole('sd5', 0.150, strengths['sd5']) # chs
    SD4 = sextupole('sd4', 0.150, strengths['sd4']) # cvs
    SF3 = sextupole('sf3', 0.150, strengths['sf3']) # chs

    # -- bpms and fast correctors --
    BPM    = marker('bpm')
    CF     = quadrupole('cf', 0.100, 0.0)

    # -- rf cavities --
    RFC = rfcavity('cav', 0, 2.5e6, 500e6)

    # -- transport lines --

    M2A = [GIRDER,CF,L11,SFA,L12,BPM,L14,QFA,L24,QDA,L15,SDA,L19,GIRDER]               # high beta xxM2 girder
    M1A = M2A[::-1]                                                                    # high beta xxM1 girder
    IDA = [GIRDER,LIA,MIDA,L50,L50,MIA,MOMACCEP,L50,L50,MIDA,LIA,GIRDER]               # high beta ID straight section
    CAV = [GIRDER,LIA,L50,L50,MIA,MOMACCEP,RFC,L50,L50,LIA,GIRDER]                     # high beta RF cavity straight section
    INJ = [GIRDER,LIA,L50,L50,END,START,MIA,MOMACCEP,L50,L50,LIA,GIRDER]               # high beta INJ straight section
    M1B = [GIRDER,L19,SDB,L15,QDB1,L24,QFB,L14,BPM,L12,SFB,L11,CF,L13,QDB2,GIRDER]     # low beta xxM1 girder
    M2B = M1B[::-1]                                                                    # low beta xxM2 girder
    IDB = [GIRDER,LIB,MIDB,L50,L50,MIB,MOMACCEP,L50,L50,MIDB,LIB,GIRDER]               # low beta ID straight section
    C3  = [LBC,BC,LBC]                                                                 # arc sector in between B3-B3
    C1A = [GIRDER,L61,SD1,L17,QF1,L14, BPM,L12,SF1,L23,QF2,L17,SD2,L21,BPM,L13,GIRDER] # arc sector in between B1-B2 (high beta odd-numbered straight sections)
    C2A = [GIRDER,L46,SD3,L17,QF3,L23,SF2,L12,BPM,L14,QF4,L12,CF,L38,BPM,L13,GIRDER]   # arc sector in between B2-B3 (high beta odd-numbered straight sections)
    C4A = [GIRDER,L73,QF4,L14,BPM,L12,SF3,L23,QF3,L17,SD4,L11,CF,L25,GIRDER]           # arc sector in between B3-B2 (high beta odd-numbered straight sections)
    C5A = [GIRDER,L34,SD5,L17,QF2,L23,SF4,L12,BPM,L14,QF1,L17,SD6,L48,BPM,L13,GIRDER]  # arc sector in between B2-B1 (high beta odd-numbered straight sections)
    C1B = [GIRDER,L61,SD6,L17,QF1,L14,BPM,L12,SF4,L23,QF2,L17,SD5,L21,BPM,L13,GIRDER]  # arc sector in between B1-B2 (low beta even-numbered straight sections)
    C2B = [GIRDER,L46,SD4,L17,QF3,L23,SF3,L12,BPM,L14,QF4,L12,CF,L38,BPM,L13,GIRDER]   # arc sector in between B2-B3 (low beta even-numbered straight sections)
    C4B = [GIRDER,L73,QF4,L14,BPM,L12,SF2,L23,QF3,L17,SD3,L11,CF,L25,GIRDER]           # arc sector in between B3-B2 (low beta even-numbered straight sections)
    C5B = [GIRDER,L34,SD2,L17,QF2,L23,SF1,L12,BPM,L14,QF1,L17,SD1,L48,BPM,L13,GIRDER]  # arc sector in between B2-B1 (low beta even-numbered straight sections)

    # -- GIRDERS --

    # straight sections
    GIRDER_01S = INJ; GIRDER_02S = IDB;
    GIRDER_03S = CAV; GIRDER_04S = IDB;
    GIRDER_05S = IDA; GIRDER_06S = IDB;
    GIRDER_07S = IDA; GIRDER_08S = IDB;
    GIRDER_09S = IDA; GIRDER_10S = IDB;
    GIRDER_11S = IDA; GIRDER_12S = IDB;
    GIRDER_13S = IDA; GIRDER_14S = IDB;
    GIRDER_15S = IDA; GIRDER_16S = IDB;
    GIRDER_17S = IDA; GIRDER_18S = IDB;
    GIRDER_19S = IDA; GIRDER_20S = IDB;

    # down and upstream straight sections
    GIRDER_01M1 = M1A; GIRDER_01M2 = M2A; GIRDER_02M1 = M1B; GIRDER_02M2 = M2B;
    GIRDER_03M1 = M1A; GIRDER_03M2 = M2A; GIRDER_04M1 = M1B; GIRDER_04M2 = M2B;
    GIRDER_05M1 = M1A; GIRDER_05M2 = M2A; GIRDER_06M1 = M1B; GIRDER_06M2 = M2B;
    GIRDER_07M1 = M1A; GIRDER_07M2 = M2A; GIRDER_08M1 = M1B; GIRDER_08M2 = M2B;
    GIRDER_09M1 = M1A; GIRDER_09M2 = M2A; GIRDER_10M1 = M1B; GIRDER_10M2 = M2B;
    GIRDER_11M1 = M1A; GIRDER_11M2 = M2A; GIRDER_12M1 = M1B; GIRDER_12M2 = M2B;
    GIRDER_13M1 = M1A; GIRDER_13M2 = M2A; GIRDER_14M1 = M1B; GIRDER_14M2 = M2B;
    GIRDER_15M1 = M1A; GIRDER_15M2 = M2A; GIRDER_16M1 = M1B; GIRDER_16M2 = M2B;
    GIRDER_17M1 = M1A; GIRDER_17M2 = M2A; GIRDER_18M1 = M1B; GIRDER_18M2 = M2B;
    GIRDER_19M1 = M1A; GIRDER_19M2 = M2A; GIRDER_20M1 = M1B; GIRDER_20M2 = M2B;

    # dispersive arcs
    GIRDER_01C1 = C1A; GIRDER_01C2 = C2A; GIRDER_01C3 = C3; GIRDER_01C4 = C4A; GIRDER_01C5 = C5A;
    GIRDER_02C1 = C1B; GIRDER_02C2 = C2B; GIRDER_02C3 = C3; GIRDER_02C4 = C4B; GIRDER_02C5 = C5B;
    GIRDER_03C1 = C1A; GIRDER_03C2 = C2A; GIRDER_03C3 = C3; GIRDER_03C4 = C4A; GIRDER_03C5 = C5A;
    GIRDER_04C1 = C1B; GIRDER_04C2 = C2B; GIRDER_04C3 = C3; GIRDER_04C4 = C4B; GIRDER_04C5 = C5B;
    GIRDER_05C1 = C1A; GIRDER_05C2 = C2A; GIRDER_05C3 = C3; GIRDER_05C4 = C4A; GIRDER_05C5 = C5A;
    GIRDER_06C1 = C1B; GIRDER_06C2 = C2B; GIRDER_06C3 = C3; GIRDER_06C4 = C4B; GIRDER_06C5 = C5B;
    GIRDER_07C1 = C1A; GIRDER_07C2 = C2A; GIRDER_07C3 = C3; GIRDER_07C4 = C4A; GIRDER_07C5 = C5A;
    GIRDER_08C1 = C1B; GIRDER_08C2 = C2B; GIRDER_08C3 = C3; GIRDER_08C4 = C4B; GIRDER_08C5 = C5B;
    GIRDER_09C1 = C1A; GIRDER_09C2 = C2A; GIRDER_09C3 = C3; GIRDER_09C4 = C4A; GIRDER_09C5 = C5A;
    GIRDER_10C1 = C1B; GIRDER_10C2 = C2B; GIRDER_10C3 = C3; GIRDER_10C4 = C4B; GIRDER_10C5 = C5B;
    GIRDER_11C1 = C1A; GIRDER_11C2 = C2A; GIRDER_11C3 = C3; GIRDER_11C4 = C4A; GIRDER_11C5 = C5A;
    GIRDER_12C1 = C1B; GIRDER_12C2 = C2B; GIRDER_12C3 = C3; GIRDER_12C4 = C4B; GIRDER_12C5 = C5B;
    GIRDER_13C1 = C1A; GIRDER_13C2 = C2A; GIRDER_13C3 = C3; GIRDER_13C4 = C4A; GIRDER_13C5 = C5A;
    GIRDER_14C1 = C1B; GIRDER_14C2 = C2B; GIRDER_14C3 = C3; GIRDER_14C4 = C4B; GIRDER_14C5 = C5B;
    GIRDER_15C1 = C1A; GIRDER_15C2 = C2A; GIRDER_15C3 = C3; GIRDER_15C4 = C4A; GIRDER_15C5 = C5A;
    GIRDER_16C1 = C1B; GIRDER_16C2 = C2B; GIRDER_16C3 = C3; GIRDER_16C4 = C4B; GIRDER_16C5 = C5B;
    GIRDER_17C1 = C1A; GIRDER_17C2 = C2A; GIRDER_17C3 = C3; GIRDER_17C4 = C4A; GIRDER_17C5 = C5A;
    GIRDER_18C1 = C1B; GIRDER_18C2 = C2B; GIRDER_18C3 = C3; GIRDER_18C4 = C4B; GIRDER_18C5 = C5B;
    GIRDER_19C1 = C1A; GIRDER_19C2 = C2A; GIRDER_19C3 = C3; GIRDER_19C4 = C4A; GIRDER_19C5 = C5A;
    GIRDER_20C1 = C1B; GIRDER_20C2 = C2B; GIRDER_20C3 = C3; GIRDER_20C4 = C4B; GIRDER_20C5 = C5B;


    # SECTORS # 01..20
    S01 = [GIRDER_01M1, GIRDER_01S, GIRDER_01M2, B1, GIRDER_01C1, B2, GIRDER_01C2, B3, GIRDER_01C3, B3, GIRDER_01C4, B2, GIRDER_01C5, B1];
    S02 = [GIRDER_02M1, GIRDER_02S, GIRDER_02M2, B1, GIRDER_02C1, B2, GIRDER_02C2, B3, GIRDER_02C3, B3, GIRDER_02C4, B2, GIRDER_02C5, B1];
    S03 = [GIRDER_03M1, GIRDER_03S, GIRDER_03M2, B1, GIRDER_03C1, B2, GIRDER_03C2, B3, GIRDER_03C3, B3, GIRDER_03C4, B2, GIRDER_03C5, B1];
    S04 = [GIRDER_04M1, GIRDER_04S, GIRDER_04M2, B1, GIRDER_04C1, B2, GIRDER_04C2, B3, GIRDER_04C3, B3, GIRDER_04C4, B2, GIRDER_04C5, B1];
    S05 = [GIRDER_05M1, GIRDER_05S, GIRDER_05M2, B1, GIRDER_05C1, B2, GIRDER_05C2, B3, GIRDER_05C3, B3, GIRDER_05C4, B2, GIRDER_05C5, B1];
    S06 = [GIRDER_06M1, GIRDER_06S, GIRDER_06M2, B1, GIRDER_06C1, B2, GIRDER_06C2, B3, GIRDER_06C3, B3, GIRDER_06C4, B2, GIRDER_06C5, B1];
    S07 = [GIRDER_07M1, GIRDER_07S, GIRDER_07M2, B1, GIRDER_07C1, B2, GIRDER_07C2, B3, GIRDER_07C3, B3, GIRDER_07C4, B2, GIRDER_07C5, B1];
    S08 = [GIRDER_08M1, GIRDER_08S, GIRDER_08M2, B1, GIRDER_08C1, B2, GIRDER_08C2, B3, GIRDER_08C3, B3, GIRDER_08C4, B2, GIRDER_08C5, B1];
    S09 = [GIRDER_09M1, GIRDER_09S, GIRDER_09M2, B1, GIRDER_09C1, B2, GIRDER_09C2, B3, GIRDER_09C3, B3, GIRDER_09C4, B2, GIRDER_09C5, B1];
    S10 = [GIRDER_10M1, GIRDER_10S, GIRDER_10M2, B1, GIRDER_10C1, B2, GIRDER_10C2, B3, GIRDER_10C3, B3, GIRDER_10C4, B2, GIRDER_10C5, B1];
    S11 = [GIRDER_11M1, GIRDER_11S, GIRDER_11M2, B1, GIRDER_11C1, B2, GIRDER_11C2, B3, GIRDER_11C3, B3, GIRDER_11C4, B2, GIRDER_11C5, B1];
    S12 = [GIRDER_12M1, GIRDER_12S, GIRDER_12M2, B1, GIRDER_12C1, B2, GIRDER_12C2, B3, GIRDER_12C3, B3, GIRDER_12C4, B2, GIRDER_12C5, B1];
    S13 = [GIRDER_13M1, GIRDER_13S, GIRDER_13M2, B1, GIRDER_13C1, B2, GIRDER_13C2, B3, GIRDER_13C3, B3, GIRDER_13C4, B2, GIRDER_13C5, B1];
    S14 = [GIRDER_14M1, GIRDER_14S, GIRDER_14M2, B1, GIRDER_14C1, B2, GIRDER_14C2, B3, GIRDER_14C3, B3, GIRDER_14C4, B2, GIRDER_14C5, B1];
    S15 = [GIRDER_15M1, GIRDER_15S, GIRDER_15M2, B1, GIRDER_15C1, B2, GIRDER_15C2, B3, GIRDER_15C3, B3, GIRDER_15C4, B2, GIRDER_15C5, B1];
    S16 = [GIRDER_16M1, GIRDER_16S, GIRDER_16M2, B1, GIRDER_16C1, B2, GIRDER_16C2, B3, GIRDER_16C3, B3, GIRDER_16C4, B2, GIRDER_16C5, B1];
    S17 = [GIRDER_17M1, GIRDER_17S, GIRDER_17M2, B1, GIRDER_17C1, B2, GIRDER_17C2, B3, GIRDER_17C3, B3, GIRDER_17C4, B2, GIRDER_17C5, B1];
    S18 = [GIRDER_18M1, GIRDER_18S, GIRDER_18M2, B1, GIRDER_18C1, B2, GIRDER_18C2, B3, GIRDER_18C3, B3, GIRDER_18C4, B2, GIRDER_18C5, B1];
    S19 = [GIRDER_19M1, GIRDER_19S, GIRDER_19M2, B1, GIRDER_19C1, B2, GIRDER_19C2, B3, GIRDER_19C3, B3, GIRDER_19C4, B2, GIRDER_19C5, B1];
    S20 = [GIRDER_20M1, GIRDER_20S, GIRDER_20M2, B1, GIRDER_20C1, B2, GIRDER_20C2, B3, GIRDER_20C3, B3, GIRDER_20C4, B2, GIRDER_20C5, B1];

    anel = [S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20];
    the_ring = _pyaccel.lattice.build(anel)

    # -- shifts model to marker 'start'
    idx = _pyaccel.lattice.find_indices(the_ring, 'fam_name', 'start')
    the_ring = _pyaccel.lattice.shift(the_ring, idx[0])

    # -- sets rf frequency
    set_rf_frequency(the_ring)

    # -- sets number of integration steps
    set_num_integ_steps(the_ring)

    return the_ring

def set_rf_frequency(the_ring):

    circumference = _pyaccel.lattice.length(the_ring)
    velocity = _mp.constants.light_speed
    rev_frequency = velocity / circumference
    rf_frequency  = _harmonic_number * rev_frequency
    idx = _pyaccel.lattice.find_indices(the_ring, 'fam_name', 'cav')
    for i in idx:
        the_ring[i].frequency = rf_frequency

def set_num_integ_steps(the_ring):

    len_bends = 0.050
    len_quads = 0.015
    len_sexts = 0.015
    for i in range(len(the_ring)):
        if the_ring[i].angle:
            nr_steps = int(_math.ceil(the_ring[i].length/len_bends))
            the_ring[i].nr_steps = nr_steps
        elif the_ring[i].polynom_b[1]:
            nr_steps = int(_math.ceil(the_ring[i].length/len_quads))
            the_ring[i].nr_steps = nr_steps
        elif the_ring[i].polynom_b[2]:
            nr_steps = int(_math.ceil(the_ring[i].length/len_sexts))
            the_ring[i].nr_steps = nr_steps
