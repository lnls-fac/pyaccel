
import os
import unittest
import numpy
import pyaccel
import trackcpp
import models

class TestLattice(unittest.TestCase):

    def setUp(self):
        self.element = pyaccel.elements.Element()
        self.the_ring = models.create_accelerator()

    def tearDown(self):
        pass

    def test_length(self):
        length = pyaccel.lattice.length(self.the_ring)
        self.assertAlmostEqual(length, 518.396)

    def test_find_spos(self):
        s = [0, 0, 0, 0, 0.5000, 1.0000, 3.4129, 3.4129,
            3.4129, 3.5129, 3.6229, 3.7729, 3.8929, 3.8929,
            4.0329, 4.2329, 4.4729, 4.6129, 4.7629, 4.9129]

        indices = [i for i in range(20)]
        pos = pyaccel.lattice.find_spos(self.the_ring, indices)
        for i in range(20):
            self.assertAlmostEqual(pos[i], s[i])

        position = pyaccel.lattice.find_spos(self.the_ring)
        ind = len(self.the_ring) - 1
        self.assertAlmostEqual(position[ind], 518.3960)

    def test_flatten(self):
        flat_elist = [self.element]*12
        elist = [self.element,self.element,[self.element,[self.element,
                [[self.element,self.element],self.element],self.element],
                self.element,[self.element,[self.element,]],self.element]]
        elist = pyaccel.lattice.flatten(elist)
        self.assertEqual(elist, flat_elist)

    def test_build(self):
        elist = [self.element,self.element,[self.element,[self.element,
                [[self.element,self.element],self.element],self.element],
                self.element,[self.element,[self.element,]],self.element]]
        lattice = pyaccel.lattice.build(elist)
        self.assertTrue(isinstance(lattice, trackcpp.trackcpp.CppElementVector))

    def test_shift(self):
        lattice = [e for e in self.the_ring]
        fam_name = 'end'
        start = len(lattice) - 1
        lattice = pyaccel.lattice.shift(lattice, start)
        self.assertEqual(len(lattice), len(self.the_ring))
        self.assertEqual(lattice[0].fam_name, fam_name)

    def test_find_indices(self):
        indices_bc = pyaccel.lattice.find_indices(self.the_ring, 'polynom_b', [0, -0.0001586, -28.62886])
        for i in indices_bc:
            self.assertEqual(self.the_ring[i].fam_name,'bc')

        mia = [1, 327, 655, 983, 1311, 1639, 1967, 2295, 2623, 2951]
        indices_mia = pyaccel.lattice.find_indices(self.the_ring, 'fam_name', 'mia')
        for i in range(len(mia)):
            self.assertEqual(indices_mia[i], mia[i])

    def test_get_attribute(self):
        length = pyaccel.lattice.get_attribute(self.the_ring, 'length')
        self.assertAlmostEqual(sum(length),518.396)

        fam_name = pyaccel.lattice.get_attribute(self.the_ring, 'fam_name', range(20))
        for i in range(20):
            self.assertEqual(fam_name[i], self.the_ring[i].fam_name)

        polynom_b = pyaccel.lattice.get_attribute(self.the_ring, 'polynom_b', range(20), m=1)
        for i in range(20):
            self.assertEqual(polynom_b[i],self.the_ring[i].polynom_b[1])

        r_in = pyaccel.lattice.get_attribute(self.the_ring,'r_in',range(20), m=1, n=1)
        for i in range(20):
            self.assertEqual(r_in[i],self.the_ring[i].r_in[1,1])

    def test_set_attribute(self):
        pyaccel.lattice.set_attribute(self.the_ring, 'length', 1, 1)
        self.assertEqual(self.the_ring[1].length, 1)

        pyaccel.lattice.set_attribute(self.the_ring, 'fam_name', [1, 2], ['test1', 'test2'])
        self.assertEqual(self.the_ring[1].fam_name, 'test1')
        self.assertEqual(self.the_ring[2].fam_name, 'test2')

        pyaccel.lattice.set_attribute(self.the_ring, 'polynom_b', [1,2], [[1,1,1], [2,2,2]])
        self.assertEqual(self.the_ring[1].polynom_b[0], 1)
        self.assertEqual(self.the_ring[2].polynom_b[0], 2)

        pyaccel.lattice.set_attribute(self.the_ring, 'polynom_b', [1,2], [[1,1,1]])
        self.assertEqual(self.the_ring[1].polynom_b[0], 1)
        self.assertEqual(self.the_ring[2].polynom_b[0], 1)

        pyaccel.lattice.set_attribute(self.the_ring, 'r_in', 1, [numpy.zeros((6,6))])
        self.assertEqual(self.the_ring[1].r_in[0,0], 0)

    def test_find_dict(self):
        names_dict=pyaccel.lattice.find_dict(self.the_ring, 'fam_name')
        for key in names_dict.keys():
            ind=names_dict[key]
            for i in ind:
                self.assertEqual(self.the_ring[i].fam_name, key)


class TestFlatFile(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(pyaccel.__path__[0], '..', 'test')
        filename = os.path.join(self.test_dir, 'flatfile.txt')
        self.a = pyaccel.lattice.read_flat_file(filename)

    def test_read_flat_file(self):
        # Accelerator fields
        self.assertAlmostEqual(self.a.energy, 3.0e9, 9)
        self.assertEqual(self.a.harmonic_number, 864)
        self.assertTrue(self.a.cavity_on)
        self.assertFalse(self.a.radiation_on)
        self.assertFalse(self.a.vchamber_on)
        # Lattice elements
        self.assertEqual(self.a[0].fam_name, 'start')
        self.assertEqual(self.a[1].fam_name, 'l50')
        self.assertAlmostEqual(self.a[1].length, +5.00000000000000000E-01, 16)
        self.assertAlmostEqual(self.a[1].hmax, +1.17000000000000003E-02, 16)
        self.assertAlmostEqual(self.a[1].vmax, +1.17000000000000003E-02, 16)
        self.assertEqual(self.a[2].pass_method, 'identity_pass')

    def test_write_flat_file(self):
        t = numpy.array([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6, 6.0e-6])
        self.a.energy = 1.5e9
        self.a[1].t_in = t
        self.a[1].t_out = -t
        filename = os.path.join(self.test_dir, 'flatfile2.txt')
        pyaccel.lattice.write_flat_file(self.a, filename)
        a = pyaccel.lattice.read_flat_file(filename)

        # Accelerator fields
        self.assertAlmostEqual(self.a.energy, 1.5e9, 9)
        self.assertEqual(self.a.harmonic_number, 864)
        self.assertTrue(self.a.cavity_on)
        self.assertFalse(self.a.radiation_on)
        self.assertFalse(self.a.vchamber_on)
        # Lattice elements
        self.assertEqual(self.a[0].fam_name, 'start')
        self.assertTrue((a[1].t_in == t).all())
        self.assertTrue((a[1].t_out == -t).all())


def lattice_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLattice)
    return suite


def flat_file_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlatFile)
    return suite


def get_suite():
    suite_list = []
    suite_list.append(lattice_suite())
    suite_list.append(flat_file_suite())
    return unittest.TestSuite(suite_list)
