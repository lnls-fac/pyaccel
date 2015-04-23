
import unittest
import numpy
import pyaccel
import trackcpp
import sirius


class TestTwiss(unittest.TestCase):

    def setUp(self):
        self.accelerator = sirius.create_accelerator()
        pyaccel.tracking.set6dtracking(self.accelerator)
        twiss = pyaccel.optics.calctwiss(self.accelerator)

        (self.spos, #self.closed_orbit,
         self.mux, self.betax, self.alphax, self.etax, self.etaxl,
         self.muy, self.betay, self.alphay, self.etay, self.etayl) = \
        pyaccel.optics.gettwiss(twiss,
                ('spos', #'closed_orbit',
                 'mux', 'betax', 'alphax', 'etax', 'etaxl',
                 'muy', 'betay', 'alphay', 'etay', 'etayl')
        )

    def test_twiss(self):
        # Test length
        self.assertEqual(len(self.betax), len(self.accelerator))

        # Values from AT
        indices = [0, 100, 200, 300, 400, 1000, 2000, 3000, 3278]
        spos = [
            0.0,
            14.307900000000007,
            33.120700000000014,
            45.778699999999994,
            64.606500000000011,
            1.601317000000002e+02,
            3.182384999999976e+02,
            4.759379666666598e+02,
            5.183959999999920e+02
        ]
        mux = [
            0.0,
            8.331616558882530,
            19.142256173386539,
            27.496143233245871,
            36.812970756086862,
            90.997762480926795,
            1.844875567009963e+02,
            2.766393254117137e+02,
            3.024146897475059e+02
        ]
        betax = [
            17.682277079604876,
            6.701209670257165,
            10.568368568858389,
            1.151432866868920,
            0.460755688636370,
            6.834409545835888,
            10.588865998200530,
            0.291064286526706,
            17.682277079526866
        ]
        alphax = [
            -1.374902365903863e-10,
            -4.858852929502183,
            -1.421999227536901,
            2.336875978595448,
            0.543209003222551,
            6.642015311944500,
            -1.425163481628221,
            -0.082999862167238,
            -1.374902960395609e-10
        ]
        etax = [
            4.596345052457250e-06,
            0.047437601563607,
            0.076591845860430,
            0.027630209056788,
            0.006143455783628,
            2.751451648532713e-06,
            0.076580184588520,
            0.016293300052488,
            4.596345056854225e-06
        ]
        etaxl = [
            -2.301864914194596e-17,
            0.041070028299318,
            0.006884952484454,
            -0.052483873706051,
            -0.012811604133997,
            -2.786871261255354e-06,
            0.006884278617381,
            0.004626392196861,
            -2.847946870637948e-16
        ]
        muy = [
            0.0,
            2.140138168772139,
            5.587753252902786,
            7.274880350820780,
            10.069951787685508,
            25.628577540164926,
            50.514629160783315,
            75.405383116994344,
            82.412591757380184
        ]
        betay = [
            3.507767173981167,
            3.194855818617335,
            7.919302652697812,
            20.263428306220863,
            6.174542678178565,
            18.133748177131761,
            7.929407865834737,
            26.949737332254749,
            3.507767174023039
        ]
        alphay = [
            8.684672140391149e-12,
            1.218379508962949,
            0.125515247190113,
            -8.415823112083245,
            0.026261768454975,
            -5.966114294365506,
            0.125655454716279,
            4.755324118976020,
            8.684495297762399e-12
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
        etayl = [
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


        # Test spos
        for i, x in zip(indices, spos):
            self.assertAlmostEqual(self.spos[i], x)

        # Test mux
        for i, x in zip(indices, mux):
            self.assertAlmostEqual(self.mux[i], x, 2)

        # Test betax
        for i, x in zip(indices, betax):
            self.assertAlmostEqual(self.betax[i], x, 1)

        # Test alphax
        for i, x in zip(indices, alphax):
            self.assertAlmostEqual(self.alphax[i], x, 1)

        # Test etax
        for i, x in zip(indices, etax):
            self.assertAlmostEqual(self.etax[i], x, 1)

        # Test etaxl
        for i, x in zip(indices, etaxl):
            self.assertAlmostEqual(self.etaxl[i], x, 1)

        # Test muy
        for i, x in zip(indices, muy):
            self.assertAlmostEqual(self.muy[i], x, 2)

        # Test betay
        for i, x in zip(indices, betay):
            self.assertAlmostEqual(self.betay[i], x, 1)

        # Test alphay
        for i, x in zip(indices, alphay):
            self.assertAlmostEqual(self.alphay[i], x, 1)

        # Test etay
        for i, x in zip(indices, etay):
            self.assertAlmostEqual(self.etay[i], x, 1)

        # Test etayl
        for i, x in zip(indices, etayl):
            self.assertAlmostEqual(self.etayl[i], x, 1)


class TestOptics(unittest.TestCase):

    def setUp(self):
        self.accelerator = sirius.create_accelerator()

    def test_getrffrequency(self):
        f = pyaccel.optics.getrffrequency(self.accelerator)
        self.assertAlmostEqual(f, 499657944.8037381, 7)

    def test_getrevolutionperiod(self):
        t = pyaccel.optics.getrevolutionperiod(self.accelerator)
        self.assertAlmostEqual(t, 1.7291829520280572e-06, 15)

    def test_getrevolutionfrequency(self):
        f = pyaccel.optics.getrevolutionfrequency(self.accelerator)
        self.assertAlmostEqual(f, 1.0/1.7291829520280572e-06, 15)

    def test_getfractunes(self):
        pyaccel.tracking.set6dtracking(self.accelerator)
        tunes = pyaccel.optics.getfractunes(self.accelerator)
        self.assertAlmostEqual(tunes[0], 0.13096034795224765, 15)
        self.assertAlmostEqual(tunes[1], 0.11652601831885964, 15)


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
