
import unittest
import numpy
import pyaccel
import trackcpp
import sirius


class TestTwiss(unittest.TestCase):

    def setUp(self):
        self.accelerator = sirius.create_accelerator()
        self.accelerator.cavity_on = True
        self.accelerator.radiation_on = False
        twiss, *_ = pyaccel.optics.calc_twiss(self.accelerator)
        (self.mux, self.betax, self.alphax, self.etax, self.etapx,
         self.muy, self.betay, self.alphay, self.etay, self.etapy) = \
        (twiss.mux, twiss.betax, twiss.alphax, twiss.etax, twiss.etapx,
        twiss.muy, twiss.betay, twiss.alphay, twiss.etay, twiss.etapy)


    def test_twiss(self):
        tol = 1.0e-8

        # Test length
        self.assertEqual(len(self.betax), len(self.accelerator))

        # Values from AT
        indices = [0, 100, 200, 300, 400, 1000, 2000, 3000, 3278]
        mux = [
            0.000000000000000e+00,
            8.331616558882587e+00,
            1.914225617338654e+01,
            2.749614323324591e+01,
            3.681297075611506e+01,
            9.099776248092682e+01,
            1.844875567009279e+02,
            2.766393254114925e+02,
            3.024146897475059e+02
        ]
        betax = [
            1.768227707960537e+01,
            6.701209670257366e+00,
            1.056836856885831e+01,
            1.151432866868986e+00,
            4.607556887205947e-01,
            6.834409545835745e+00,
            1.058886599313864e+01,
            2.910642866605170e-01,
            1.768227707952670e+01
        ]
        alphax = [
            -1.374336922729196e-10,
            -4.858852929502383e+00,
            -1.421999227536827e+00,
            2.336875978595554e+00,
            5.432090033318180e-01,
            6.642015311944366e+00,
            -1.425163480991599e+00,
            -8.299986198405727e-02,
            -1.374337372097140e-10
        ]
        etax = [
            4.595731534213989e-06,
            4.743760196046474e-02,
            7.659184549737912e-02,
            2.763020924143351e-02,
            6.142987795112390e-03,
            2.751543333620040e-06,
            7.657435199079558e-02,
            1.629205898120782e-02,
            4.595731539083780e-06
        ]
        etapx = [
            2.309267029925114e-11,
            4.107002861987864e-02,
            6.884952477154262e-03,
            -5.248387404979547e-02,
            -1.281062835243484e-02,
            -2.786954763147480e-06,
            6.883754323651932e-03,
            4.626039875944818e-03,
            2.309263966924268e-11
        ]
        muy = [
            0.000000000000000e+00,
            2.140138168772155e+00,
            5.587753252902800e+00,
            7.274880350820775e+00,
            1.006995178768550e+01,
            2.562857754016492e+01,
            5.051462916078330e+01,
            7.540538311699437e+01,
            8.241259175738020e+01
        ]
        betay = [
            3.507767173981249e+00,
            3.194855818617326e+00,
            7.919302652697906e+00,
            2.026342830622053e+01,
            6.174542678178558e+00,
            1.813374817713175e+01,
            7.929407865834472e+00,
            2.694973733225292e+01,
            3.507767174022887e+00
        ]
        alphay = [
            8.692403376538394e-12,
            1.218379508962921e+00,
            1.255152471900926e-01,
            -8.415823112083089e+00,
            2.626176845500004e-02,
            -5.966114294365477e+00,
            1.256554547162735e-01,
            4.755324118975679e+00,
            8.692407902629905e-12
        ]
        etay = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]
        etapy = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]

        # Test mux
        for i, x in zip(indices, mux):
            self.assertAlmostEqual(self.mux[i], x, 8)

        # Test betax
        for i, x in zip(indices, betax):
            diff = (self.betax[i] - x)/x
            self.assertAlmostEqual(diff, 0.0, 3)
            # self.assertAlmostEqual(self.betax[i], x, 8)

        # Test alphax
        for i, x in zip(indices, alphax):
            if x > tol:
                diff = (self.alphax[i] - x)/x
            else:
                diff = self.alphax[i] - x
            self.assertAlmostEqual(diff, 0.0, 3)
            # self.assertAlmostEqual(self.alphax[i], x, 8)

        # Test etax
        for i, x in zip(indices, etax):
            if x > 1.0e-4:
                diff = (self.etax[i] - x)/x
            else:
                diff = self.etax[i] - x
            self.assertAlmostEqual(diff, 0.0, 3)
            # self.assertAlmostEqual(self.etax[i], x, 8)

        # Test etapx
        for i, x in zip(indices, etapx):
            if x > tol:
                diff = (self.etapx[i] - x)/x
            else:
                diff = self.etapx[i] - x
            self.assertAlmostEqual(diff, 0.0, 3)
            # self.assertAlmostEqual(self.etapx[i], x, 8)

        # Test muy
        for i, x in zip(indices, muy):
            self.assertAlmostEqual(self.muy[i], x, 8)

        # Test betay
        for i, x in zip(indices, betay):
            diff = (self.betay[i] - x)/x
            self.assertAlmostEqual(diff, 0.0, 3)
            # self.assertAlmostEqual(self.betay[i], x, 8)

        # Test alphay
        for i, x in zip(indices, alphay):
            if x > tol:
                diff = (self.alphay[i] - x)/x
            else:
                diff = self.alphay[i] - x
            self.assertAlmostEqual(diff, 0.0, 3)
            # self.assertAlmostEqual(self.alphay[i], x, 8)

        # Test etay
        for i, x in zip(indices, etay):
            self.assertAlmostEqual(self.etay[i], x, 8)

        # Test etapy
        for i, x in zip(indices, etapy):
            self.assertAlmostEqual(self.etapy[i], x, 8)


class TestOptics(unittest.TestCase):

    def setUp(self):
        self.accelerator = sirius.create_accelerator()

    def test_get_rf_frequency(self):
        f = pyaccel.optics.get_rf_frequency(self.accelerator)
        self.assertAlmostEqual(f, 4.996579520521069e+08, 6)

    def test_get_revolution_period(self):
        t = pyaccel.optics.get_revolution_period(self.accelerator)
        self.assertAlmostEqual(t, 1.7291829520280572e-06, 15)

    def test_get_revolution_frequency(self):
        f = pyaccel.optics.get_revolution_frequency(self.accelerator)
        self.assertAlmostEqual(f, 1.0/1.7291829520280572e-06, 15)

    def test_get_frac_tunes(self):
        self.accelerator.cavity_on = True
        self.accelerator.radiation_on = False
        tunes = pyaccel.optics.get_frac_tunes(self.accelerator)
        self.assertAlmostEqual(tunes[0], 0.130792736910679, 10)
        self.assertAlmostEqual(tunes[1], 0.116371351207661, 10)


def twiss_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTwiss)
    return suite


def optics_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptics)
    return suite


def get_suite():
    suite_list = []
    suite_list.append(twiss_suite())
    suite_list.append(optics_suite())
    return unittest.TestSuite(suite_list)
