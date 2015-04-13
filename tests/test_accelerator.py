
import unittest
import numpy
import pyaccel
import trackcpp
import sirius

class TestAccelerator(unittest.TestCase):

    def setUp(self):
        self.the_ring = sirius.create_accelerator()

    def tearDown(self):
        pass

    def test_energy(self):
        self.assertEqual(self.the_ring.energy, 3e9)
    def test_harmonic_number(self):
        self.assertEqual(self.the_ring.harmonic_number, 864)
    def test_cavity_on(self):
        self.assertFalse(self.the_ring.cavity_on)
    def test_radiation_on(self):
        self.assertFalse(self.the_ring.radiation_on)
    def test_vchamber_on(self):
        self.assertFalse(self.the_ring.vchamber_on)
    def test_number_of_elements(self):
        self.assertEqual(len(self.the_ring), 3279)
    def test_length(self):
        self.assertAlmostEqual(self.the_ring.length, 518.396, places=10)
    def test_set_basic_parameters(self):
        self.the_ring.harmonic_number = 12
        self.the_ring.cavity_on = True
        self.the_ring.vchamber_on = True
        self.assertEqual(self.the_ring.harmonic_number, 12)
        self.assertTrue(self.the_ring.cavity_on)
        self.assertFalse(self.the_ring.radiation_on)
        self.assertTrue(self.the_ring.vchamber_on)
    def test_set_energy(self):
        self.the_ring.energy = 1.0e9
        self.assertAlmostEqual(self.the_ring.energy, 1.0e9, places=10)
        self.assertAlmostEqual(self.the_ring.gamma_factor, 1956.9512693314196, places=10)
        self.assertAlmostEqual(self.the_ring.beta_factor, 0.9999998694400394, places=10)
        self.assertAlmostEqual(self.the_ring.velocity, 299792418.8591085, places=10)
        self.assertAlmostEqual(self.the_ring.brho, 3.3356405164803693, places=10)
    def test_set_gamma(self):
        self.the_ring.gamma_factor = 1956.9512693314196
        self.assertAlmostEqual(self.the_ring.energy, 1.0e9, places=10)
        self.assertAlmostEqual(self.the_ring.gamma_factor, 1956.9512693314196, places=10)
        self.assertAlmostEqual(self.the_ring.beta_factor, 0.9999998694400394, places=10)
        self.assertAlmostEqual(self.the_ring.velocity, 299792418.8591085, places=10)
        self.assertAlmostEqual(self.the_ring.brho, 3.3356405164803693, places=10)
    def test_set_beta(self):
        self.the_ring.beta_factor = 0.9999998694400394
        self.assertAlmostEqual(self.the_ring.energy, 999999999.6485898, places=10)
        self.assertAlmostEqual(self.the_ring.gamma_factor, 1956.951268643727, places=10)
        self.assertAlmostEqual(self.the_ring.beta_factor, 0.9999998694400394, places=10)
        self.assertAlmostEqual(self.the_ring.velocity, 299792418.8591085, places=10)
        self.assertAlmostEqual(self.the_ring.brho, 3.335640515308191, places=10)
    def test_set_velocity(self):
        self.the_ring.velocity = 299792418.8591085
        self.assertAlmostEqual(self.the_ring.energy, 1000000000.0737672, places=10)
        self.assertAlmostEqual(self.the_ring.gamma_factor, 1956.9512694757784, places=10)
        self.assertAlmostEqual(self.the_ring.beta_factor, 0.9999998694400394, places=10)
        self.assertAlmostEqual(self.the_ring.velocity, 299792418.8591085, places=10)
        self.assertAlmostEqual(self.the_ring.brho, 3.3356405167264302, places=10)
    def test_set_brho(self):
        self.the_ring.brho = 3.3356405164803693
        self.assertAlmostEqual(self.the_ring.energy, 999999999.9999999, places=10)
        self.assertAlmostEqual(self.the_ring.gamma_factor, 1956.9512693314196, places=10)
        self.assertAlmostEqual(self.the_ring.beta_factor, 0.9999998694400394, places=10)
        self.assertAlmostEqual(self.the_ring.velocity, 299792418.8591085, places=10)
        self.assertAlmostEqual(self.the_ring.brho, 3.3356405164803693, places=10)


def accelerator_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAccelerator)
    return suite


def get_suite():
    suite_list = []
    suite_list.append(accelerator_suite())
    return unittest.TestSuite(suite_list)
