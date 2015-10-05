
import unittest
import numpy
import pyaccel
import trackcpp
import sirius


class TestTracking(unittest.TestCase):

    def setUp(self):
        self.the_ring = sirius.create_accelerator()

    def tearDown(self):
        pass

    def test_findm66(self):

        the_ring = self.the_ring
        pyaccel.tracking.set6dtracking(the_ring)
        tms = pyaccel.tracking.findm66(the_ring)

    def test_findorbit6(self):
        # find orbit without fixed point guess and indices = 'open'
        the_ring = self.the_ring
        pyaccel.tracking.set6dtracking(the_ring)
        co = pyaccel.tracking.findorbit6(the_ring, indices='open')
        self.assertAlmostEqual(sum(co[:,0]), -0.017604662816068, places=13) # 15?

        # find orbit with fixed point guess and indices = 'open'
        the_ring = self.the_ring
        pyaccel.tracking.set6dtracking(the_ring)
        fixed_point_guess = [0.002,0,0,0,0,0]
        co = pyaccel.tracking.findorbit6(the_ring, indices='open',
                fixed_point_guess=fixed_point_guess)
        self.assertEqual(co.shape[1], len(the_ring))
        self.assertAlmostEqual(sum(co[:,0]), -0.017604662816066, places=13) # 15?

        # find orbit with fixed point guess and indices = None
        the_ring = self.the_ring
        pyaccel.tracking.set6dtracking(the_ring)
        fixed_point_guess = [0.002,0,0,0,0,0]
        co = pyaccel.tracking.findorbit6(the_ring, indices=None,
                fixed_point_guess=fixed_point_guess)
        self.assertAlmostEqual(sum(co)[0], -0.017604662816066, places=13) #15?

    def test_ringpass(self):
        # one particle (python list), storing pos at end only
        the_ring = self.the_ring
        particles =[0.001,0,0,0,0,0]
        particles_out, lost_flag, lost_turn, lost_element, lost_plane = \
            pyaccel.tracking.ringpass(accelerator=the_ring,
                                      particles=particles,
                                      nr_turns=10,
                                      turn_by_turn=None)
        p1 = particles_out
        self.assertAlmostEqual(sum(p1),-3.455745799826087e-04, places=15)

        # one particle (python list), storing pos at all turns
        the_ring = self.the_ring
        particles = [0.001,0,0,0,0,0]
        particles_out, lost_flag, lost_turn, lost_element, lost_plane = \
            pyaccel.tracking.ringpass(accelerator=the_ring,
                                      particles=particles,
                                      nr_turns=100,
                                      turn_by_turn='closed')
        n1 = particles_out[:,50]
        n2 = particles_out[:,10]
        self.assertAlmostEqual(sum(n1),-0.0009014664358281, places=15)
        self.assertAlmostEqual(sum(n2),-0.0009459070222543, places=15)

        # two particles (python list), storing pos at all turns
        the_ring = self.the_ring
        particles = [[0.001,0,0,0,0,0],[0.001,0,0,0,0,0]]
        particles_out, lost_flag, lost_turn, lost_element, lost_plane = \
            pyaccel.tracking.ringpass(accelerator=the_ring,
                                      particles=particles,
                                      nr_turns=100,
                                      turn_by_turn='closed')
        n1_p1 = particles_out[0,:,50]
        n2_p1 = particles_out[0,:,10]
        n1_p2 = particles_out[1,:,50]
        n2_p2 = particles_out[1,:,10]
        self.assertAlmostEqual(sum(n1_p1),-0.0009014664358281, places=15)
        self.assertAlmostEqual(sum(n2_p1),-0.0009459070222543, places=15)
        self.assertAlmostEqual(sum(n1_p2),-0.0009014664358281, places=15)
        self.assertAlmostEqual(sum(n2_p2),-0.0009459070222543, places=15)

        # two particles (numpy array), storing pos at all turns
        the_ring = self.the_ring
        particles = numpy.zeros((2,6))
        particles[0,:] = [0.001,0,0,0,0,0]
        particles[1,:] = [0.001,0,0,0,0,0]
        particles_out, lost_flag, lost_turn, lost_element, lost_plane = \
            pyaccel.tracking.ringpass(accelerator=the_ring,
                                      particles=particles,
                                      nr_turns=120,
                                      turn_by_turn='closed')
        n1_p1 = particles_out[0,:,50]
        n2_p1 = particles_out[0,:,10]
        n1_p2 = particles_out[1,:,50]
        n2_p2 = particles_out[1,:,10]
        self.assertAlmostEqual(sum(n1_p1),-0.0009014664358281, places=15)
        self.assertAlmostEqual(sum(n2_p1),-0.0009459070222543, places=15)
        self.assertAlmostEqual(sum(n1_p2),-0.0009014664358281, places=15)
        self.assertAlmostEqual(sum(n2_p2),-0.0009459070222543, places=15)

        # one particle (python list), storing pos at all turns (begin)
        the_ring = self.the_ring
        particles = [0.001,0,0,0,0,0]
        particles_out, lost_flag, lost_turn, lost_element, lost_plane = \
            pyaccel.tracking.ringpass(accelerator=the_ring,
                                      particles=particles,
                                      nr_turns=100,
                                      turn_by_turn='open')
        p1 = particles_out[:,-1]
        self.assertAlmostEqual(sum(p1),0.0001557474602497, places=15)

    def test_linepass(self):
        #return
        # tracking of one particle through the whole line
        the_ring = self.the_ring
        particles =[0.001,0.0002,0.003,0.0004,0.005,0.006]
        particles_out, lost_flag, lost_element, lost_plane = \
            pyaccel.tracking.linepass(accelerator=the_ring,
                                      particles=particles,
                                      indices=None)
        p1 = particles_out
        self.assertAlmostEqual(sum(p1),  0.016408852801124, places=15)

        # tracking of two particles (numpy), pos at the end of line
        particles = numpy.zeros((2,6))
        particles[0,:], particles[1,:] = \
            [0.001,0.0002,0.003,0.0004,0.005,0.006], [0.020,0,0,0,0,0]
        particles_out, lost_flag, lost_element, lost_plane = \
            pyaccel.tracking.linepass(accelerator=self.the_ring,
                                      particles=particles,
                                      indices=None)
        p1, p2 = particles_out[0,:], particles_out[1,:]
        self.assertAlmostEqual(sum(p1), 0.016408852801124, places=15)
        self.assertNotEqual(sum(p2), sum(p2))
        self.assertTrue(lost_flag)
        self.assertListEqual(lost_element, [None,195])
        self.assertListEqual(lost_plane, [None, 'x'])

        # tracking of one particle (list), at all elements
        particles = [0.001,0.0002,0.003,0.0004,0.005,0.006]
        particles_out, lost_flag, lost_element, lost_plane = \
            pyaccel.tracking.linepass(accelerator=self.the_ring,
                                      particles=particles,
                                      indices='open')
        p1_e1000 = particles_out[:,1000]
        self.assertAlmostEqual(sum(p1_e1000), 0.0152805372605638, places=15)

        # tracking of two particles at two differente elemnts
        particles = numpy.zeros((2,6))
        particles[0,:] = [0.001,0.0002,0.003,0.0004,0.005,0.006]
        particles[1,:] = [0.001,-0.05,0.003,0.0004,0.005,0.006]
        particles_out, lost_flag, lost_element, lost_plane = \
            pyaccel.tracking.linepass(accelerator=self.the_ring,
                                      particles=particles,
                                      indices=[100,500])
        el1 = particles_out[:,:,0]
        el2 = particles_out[:,:,1]
        self.assertAlmostEqual(sum(el1[0,:]) , 0.0140704271348954, places=15)
        self.assertAlmostEqual(sum(el2[0,:]) , 0.0140823891384837, places=15)
        self.assertFalse(sum(el1[1,:]) == sum(el1[:,1])) # particle lost
        self.assertEqual(lost_element[1], 39)

    def test_elementpass(self):
        #return

        accelerator = {'energy':3e9,
                       'harmonic_number':864,
                       'cavity_on':False,
                       'radiation_on':False,
                       'vchamber_on':False}

        try:

            ## -- simple element tracking without radiation, cavity off --

            # tracks one particle through a driftpass
            d = pyaccel.elements.drift(fam_name='d', length=1.0)
            r = pyaccel.tracking.elementpass(element=d,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.026980049998762, places=15)

            # tracks one particle through a quadrupole
            q = pyaccel.elements.quadrupole(fam_name='q', length=1.0, K=2.0)
            r = pyaccel.tracking.elementpass(element=q,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.040352947331718, places=15)

            # tracks one particle through a sextupole
            s = pyaccel.elements.sextupole(fam_name='s', length=1.0, S=2.0)
            r = pyaccel.tracking.elementpass(element=s,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.027098408317945, places=15)

            # tracks one particle through a dipole with gradient
            d = pyaccel.elements.rbend(fam_name='d', length=1.0, angle = 0.1, angle_in=0.05, K=0.1)
            r = pyaccel.tracking.elementpass(element=d,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r),0.028318390270127, places=15)


            ## -- simple element tracking with radiation, cavity off --
            accelerator['radiation_on'] = True

            # tracks one particle through a driftpass (radiation on)
            d = pyaccel.elements.drift(fam_name='d', length=1.0)
            r = pyaccel.tracking.elementpass(element=d,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.026980049998762, places=15)

            # tracks one particle through a quadrupole (radiation_on)
            q = pyaccel.elements.quadrupole(fam_name='q', length=1.0, K=2.0)
            r = pyaccel.tracking.elementpass(element=q,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.040352869641487, places=15)

            # tracks one particle through a sextupole (radiation on)
            s = pyaccel.elements.sextupole(fam_name='s', length=1.0, S=2.0)
            r = pyaccel.tracking.elementpass(element=s,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r), 0.027098408317945, places=11)

            # tracks one particle through a dipole with gradient (radiation on)
            d = pyaccel.elements.rbend(fam_name='d', length=1.0, angle = 0.1, angle_in=0.05, K=0.1)
            r = pyaccel.tracking.elementpass(element=d,
                                             particles=[0.001,0.002,0.003,0.004,0.005,0.006],
                                             **accelerator)
            self.assertAlmostEqual(sum(r),0.028314349065038, places=11)

            # tracks one particle using a numpy pos input
            accelerator['radiation_on'] = False
            pos = numpy.zeros((1,6))
            pos[0,:] = 0.001,0.002,0.003,0.004,0.005,0.006
            q = pyaccel.elements.quadrupole(fam_name='q', length=1.0, K=2.0)
            r = pyaccel.tracking.elementpass(element=q,
                                             particles=pos,
                                             **accelerator)
            self.assertAlmostEqual(sum(r[0,:]), 0.040352947331718, places=15)


            # tracks two particles (lists) through a sextupole
            accelerator['radiation_on'] = False
            q = pyaccel.elements.quadrupole(fam_name='q', length=1.0, K=2.0)
            r = pyaccel.tracking.elementpass(element=q,
                                             particles=[
                                                [0.001,0.002,0.003,0.004,0.005,0.006],
                                                [0,0,0.001,0,0,0]],
                                             **accelerator)
            self.assertAlmostEqual(sum(r[0,:]),
                                   0.040352947331718, places=15)

        except pyaccel.tracking.TrackingException:
            self.assertTrue(False)

def tracking_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTracking)
    return suite


def get_suite():
    suite_list = []
    suite_list.append(tracking_suite())
    return unittest.TestSuite(suite_list)
