#!/usr/bin/env python3

import unittest
import test_elements
import test_accelerator
import test_tracking
import test_lattice

suite_list = []
#suite_list.append(test_elements.get_suite())
#suite_list.append(test_accelerator.get_suite())
#suite_list.append(test_lattice.get_suite())
suite_list.append(test_tracking.get_suite())


tests = unittest.TestSuite(suite_list)
unittest.TextTestRunner(verbosity=2).run(tests)
