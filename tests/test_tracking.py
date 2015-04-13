
import unittest
import numpy
import pyaccel
import trackcpp
import numpy

import sirius

class TestTracking(unittest.TestCase):

    def setUp(self):
        self.the_ring = sirius.create_accelerator()

    def tearDown(self):
        pass

    def test_linepass(self):
        the_ring = self.the_ring
        particles = [0.001,0,0.001,0,0,0]
        particles_out, lost_flag, lost_element, lost_plane = \
            pyaccel.tracking.linepass(accelerator=the_ring,
                                      particles=particles,
                                      indices=None)
        p1 = particles_out
        self.assertAlmostEqual(sum(p1), 0.0011086627714938745, places=15)

        # tracking of two particles (numpy), pos at the end of line
        particles = numpy.zeros((6,2))
        particles[:,0], particles[:,1] = [0.001,0,0,0,0,0], [0.020,0,0,0,0,0]
        particles_out, lost_flag, lost_element, lost_plane = \
            pyaccel.tracking.linepass(accelerator=self.the_ring,
                                      particles=particles,
                                      indices=None)
        p1, p2 = particles_out[:,0], particles_out[:,1]
        self.assertAlmostEqual(sum(p1), 0.00063339906961478495, places=15)
        self.assertNotEqual(sum(p2), sum(p2))
        self.assertTrue(lost_flag)
        self.assertListEqual(lost_element, [None,195])
        self.assertListEqual(lost_plane, [None, 'x'])

        # tracking of one particle (list), pos at more than one position
        particles = [0.001,0,0,0,0,0]
        particles_out, lost_flag, lost_element, lost_plane = \
            pyaccel.tracking.linepass(accelerator=self.the_ring,
                                      particles=particles,
                                      indices=[0,1])
        p1_e1 = particles_out[0,:]
        p1_e2 = particles_out[1,:]
        self.assertAlmostEqual(sum(p1_e1), 0.00063339906961478495, places=15)




    def test_elementpass(self):

        accelerator = {'energy':3e9,
                       'harmonic_number':864,
                       'cavity_on':False,
                       'radiation_on':False,
                       'vchamber_on':False}

        try:

            # tracks one particle through a driftpass
            d = pyaccel.elements.drift(fam_name='d', length=1.0)
            r = pyaccel.tracking.elementpass(element=d,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            print(r)
            self.assertAlmostEqual(sum(r), 0.040352947331718, places=12)

            # tracks one particle using a numpy pos input
            pos = numpy.zeros((6,1))
            pos[:,0] = 0.001,0,0,0,0,0
            r = pyaccel.tracking.elementpass(element=q,
                                             particles=pos,
                                             **accelerator)
            self.assertAlmostEqual(sum(r[:,0]), -0.00124045591936, places=12)

            # tracks one particle through a quadrupole
            q = pyaccel.elements.quadrupole(fam_name='q', length=1.0, K=2.0)
            r = pyaccel.tracking.elementpass(element=q,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.040352947331718, places=12)

            # tracks one particle using a numpy pos input
            pos = numpy.zeros((6,1))
            pos[:,0] = 0.001,0,0,0,0,0
            r = pyaccel.tracking.elementpass(element=q,
                                             particles=pos,
                                             **accelerator)
            self.assertAlmostEqual(sum(r[:,0]), -0.00124045591936, places=12)

            # tracks two particles (lists) through a sextupole
            s = pyaccel.elements.sextupole(fam_name='s', length=1.0, S=2.0)
            r = pyaccel.tracking.elementpass(element=s,
                                             particles=[[0.001,0,0,0,0,0],
                                                        [0,0,0.001,0,0,0]],
                                             **accelerator)
            self.assertAlmostEqual(sum(r[:,0])+sum(r[:,1]),
                                   0.0020000033337366449, places=12)

            # tracks one particle through a bending magnet with radiation on
            b = pyaccel.elements.rbend(fam_name='b', length=0.1, angle=0.1, K=0.2)
            accelerator['radiation_on'] = True
            r = pyaccel.tracking.elementpass(element=s,
                                             particles=[0.001,0.001,0.002,
                                                        0.002,0.003,0.003],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.015038971318609795, places=12)
        except pyaccel.tracking.TrackingException:
            self.assertTrue(False)

def tracking_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTracking)
    return suite


def get_suite():
    suite_list = []
    suite_list.append(tracking_suite())
    return unittest.TestSuite(suite_list)
