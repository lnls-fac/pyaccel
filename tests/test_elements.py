
import unittest
import numpy
import pyaccel
import trackcpp


class TestElement(unittest.TestCase):

    def setUp(self):
        self.element = pyaccel.elements.Element()

    def test_attributes(self):
        attributes = [
                'fam_name',
                'pass_method',
                'length',
                'nr_steps',
                'hkick',
                'vkick',
                'angle',
                'angle_in',
                'angle_out',
                'gap',
                'fint_in',
                'fint_out',
                'thin_KL',
                'thin_SL',
                'frequency',
                'voltage',
                'kicktable',
                'hmax',
                'vmax',
                'polynom_a',
                'polynom_b',
                't_in',
                't_out',
                'r_in',
                'r_out'
        ]

        for a in attributes:
            r = hasattr(self.element, a)
            self.assertTrue(r, "attribute '" + a + "' not found")

    def test_default_init(self):
        self.assertEqual(self.element.fam_name, "")
        self.assertEqual(self.element.length, 0.0)

    def test_array_sizes(self):
        self.assertEqual(len(self.element.t_in), 6)
        self.assertEqual(len(self.element.t_out), 6)
        self.assertEqual(self.element.r_in.shape, (6, 6))
        self.assertEqual(self.element.r_out.shape, (6, 6))

    def test_t_in_out(self):
        # Set entire vector
        t = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        self.element.t_in = t
        for i in range(len(t)):
            self.assertAlmostEqual(self.element.t_in[i], t[i])

        # Set single element
        self.element.t_in[0] = -10.0
        self.assertAlmostEqual(self.element.t_in[0], -10.0)

        # Set list of indices
        self.element.t_out[[1, 3, 5]] = [1.0, 3.0, 5.0]
        self.assertAlmostEqual(self.element.t_out[1], 1.0)
        self.assertAlmostEqual(self.element.t_out[3], 3.0)
        self.assertAlmostEqual(self.element.t_out[5], 5.0)

        # Set slice
        self.element.t_out[3:] = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(self.element.t_out[3], -1.0)
        self.assertAlmostEqual(self.element.t_out[4], -2.0)
        self.assertAlmostEqual(self.element.t_out[5], -3.0)

    def test_r_in_out(self):
        # Set entire matrix
        r = numpy.random.normal(size=(6, 6))
        self.element.r_in = r

        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                self.assertAlmostEqual(self.element.r_in[i, j], r[i, j])

        # Set single element
        self.element.r_in[2, 5] = -10.0
        self.assertAlmostEqual(self.element.r_in[2, 5], -10.0)

        # Set list of indices
        self.element.r_out[[2, 3, 4], [1, 3, 5]] = [1.0, 3.0, 5.0]
        self.assertAlmostEqual(self.element.r_out[2, 1], 1.0)
        self.assertAlmostEqual(self.element.r_out[3, 3], 3.0)
        self.assertAlmostEqual(self.element.r_out[4, 5], 5.0)

        # Set slice
        self.element.r_out[1, 3:] = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(self.element.r_out[1, 3], -1.0)
        self.assertAlmostEqual(self.element.r_out[1, 4], -2.0)
        self.assertAlmostEqual(self.element.r_out[1, 5], -3.0)

    def test_set_pass_method_from_index(self):
        for i in range(len(pyaccel.elements.pass_methods)):
            pass_method = pyaccel.elements.pass_methods[i]
            self.element.pass_method = i
            self.assertEqual(self.element.pass_method, pass_method)

    def test_set_pass_method_from_string(self):
        for pass_method in pyaccel.elements.pass_methods:
            self.element.pass_method = pass_method
            self.assertEqual(self.element.pass_method, pass_method)

    def test_set_invalid_pass_method_from_index(self):
        error = False
        try:
            self.element.pass_method = len(pyaccel.elements.pass_methods)
        except:
            error = True
        self.assertTrue(error, "invalid pass method set")

    def test_set_invalid_pass_method_from_string(self):
        error = False
        try:
            self.element.pass_method = 'invalid_pass_method'
        except:
            error = True
        self.assertTrue(error, "invalid pass method set")


class TestTrackCppElement(unittest.TestCase):

    def setUp(self):
        self.element = pyaccel.elements.Element()
        self.trackcpp_element = self.element._e

    def test_pass_method(self):
        index = 2
        self.element.pass_method = pyaccel.elements.pass_methods[index]
        self.assertEqual(self.trackcpp_element.pass_method, index)

    def test_fam_name(self):
        fam_name = 'FAM'
        self.element.fam_name = fam_name
        self.assertEqual(self.trackcpp_element.fam_name, fam_name)

    def test_angle(self):
        angle = 0.123456789
        self.element.angle = angle
        self.assertAlmostEqual(self.trackcpp_element.angle, angle)

    def test_t_in(self):
        value = -1.2345
        self.element.t_in[2] = value
        cpp_value = trackcpp.c_array_get(self.trackcpp_element.t_in, 2)
        self.assertAlmostEqual(cpp_value, value)

    def test_r_out(self):
        value = -1.2345
        self.element.r_out[2, 5] = value
        cpp_value = trackcpp.c_array_get(self.trackcpp_element.r_out, 2*6 + 5)
        self.assertAlmostEqual(cpp_value, value)


class TestCreationFunctions(unittest.TestCase):

    def test_marker(self):
        name = 'Marker'
        m = pyaccel.elements.marker(name)
        self.assertEqual(m.fam_name, name)

    def test_bpm(self):
        name = 'BPM'
        b = pyaccel.elements.bpm(name)
        self.assertEqual(b.fam_name, name)

    def test_drift(self):
        name = 'Drift'
        length = 1.2345
        d = pyaccel.elements.drift(name, length)
        self.assertEqual(d.fam_name, name)
        self.assertAlmostEqual(d.length, length)

    def test_hcorrector(self):
        name = 'CH'
        length = 1.2345
        kick = 0.4321
        c = pyaccel.elements.hcorrector(name, length, kick)
        self.assertEqual(c.fam_name, name)
        self.assertAlmostEqual(c.length, length)
        self.assertAlmostEqual(c.hkick, kick)

    def test_vcorrector(self):
        name = 'CV'
        length = 1.2345
        kick = 0.4321
        c = pyaccel.elements.vcorrector(name, length, kick)
        self.assertEqual(c.fam_name, name)
        self.assertAlmostEqual(c.length, length)
        self.assertAlmostEqual(c.vkick, kick)

    def test_corrector(self):
        name = 'Corrector'
        length = 1.2345
        hkick = 0.4321
        vkick = -0.1234
        c = pyaccel.elements.corrector(name, length, hkick, vkick)
        self.assertEqual(c.fam_name, name)
        self.assertAlmostEqual(c.length, length)
        self.assertAlmostEqual(c.hkick, hkick)
        self.assertAlmostEqual(c.vkick, vkick)

    def test_rbend(self):
        name = 'Bend'
        length = 1.2345
        angle = 5.412
        angle_in = 1.0
        angle_out = 2.0
        fint_in = 0.1
        fint_out = 0.2
        K = 1.1
        S = 2.2
        b = pyaccel.elements.rbend(
                fam_name=name,
                length=length,
                angle=angle,
                angle_in=angle_in,
                angle_out=angle_out,
                fint_in=fint_in,
                fint_out=fint_out,
                K=K,
                S=S
        )
        self.assertEqual(b.fam_name, name)
        self.assertAlmostEqual(b.length, length)
        self.assertAlmostEqual(b.angle, angle)
        self.assertAlmostEqual(b.angle_in, angle_in)
        self.assertAlmostEqual(b.angle_out, angle_out)
        self.assertAlmostEqual(b.fint_in, fint_in)
        self.assertAlmostEqual(b.fint_out, fint_out)
        self.assertAlmostEqual(b.polynom_b[1], K)
        self.assertAlmostEqual(b.polynom_b[2], S)

    def test_quadrupole(self):
        name = 'Quadrupole'
        length = 1.2345
        K = 1.1
        nr_steps = 20
        q = pyaccel.elements.quadrupole(name, length, K, nr_steps)
        self.assertEqual(q.fam_name, name)
        self.assertAlmostEqual(q.length, length)
        self.assertAlmostEqual(q.polynom_b[1], K)
        self.assertEqual(q.nr_steps, nr_steps)

    def test_sextupole(self):
        name = 'Sextupole'
        length = 1.2345
        S = 1.1
        nr_steps = 15
        s = pyaccel.elements.sextupole(name, length, S, nr_steps)
        self.assertEqual(s.fam_name, name)
        self.assertAlmostEqual(s.length, length)
        self.assertAlmostEqual(s.polynom_b[2], S)
        self.assertEqual(s.nr_steps, nr_steps)

    def test_rfcavity(self):
        name = 'RF'
        length = 1.2
        voltage = 2.5e6
        frequency = 500e6
        c = pyaccel.elements.rfcavity(name, length, voltage, frequency)
        self.assertEqual(c.fam_name, name)
        self.assertAlmostEqual(c.length, length)
        self.assertAlmostEqual(c.voltage, voltage)
        self.assertAlmostEqual(c.frequency, frequency)


def element_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestElement)
    return suite


def trackcpp_element_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackCppElement)
    return suite


def creation_functions_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCreationFunctions)
    return suite


def get_suite():
    suite_list = []
    suite_list.append(element_suite())
    suite_list.append(trackcpp_element_suite())
    suite_list.append(creation_functions_suite())
    return unittest.TestSuite(suite_list)
