#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from mpes import utils as u
import unittest as unit
import numpy as np


class TestUtils(unit.TestCase):
    """ Class for unit testing of functions in the mpes.utils module
    """
    
    def test_dictmerge(self):
        """ Testing the dictionary merging function in four scenarios.
        """
        
        carray = np.zeros((2, 3))
        
        Da = {'a':5}
        Da_alt = {'a':6}
        Db = {'b':[1, 2, 3]}
        Dc = {'c':carray}
        D_merged = {'a':5, 'b':[1, 2, 3], 'c':carray}
        
        D_dict = {'b':[1, 2, 3], 'c':carray}
        D_tuple = (Da, Db, Dc) # incl. a repeated part
        D_list = [Db, Dc]
        
        # Merge a dictionary with another dictionary
        self.assertEqual(u.dictmerge(Da, D_dict), D_merged)
        # Merge a dictionary with another dictionary incl. update
        self.assertEqual(u.dictmerge(Da, Da_alt), Da_alt)
        # Merge a dictionary with a tuple of dictionaries
        self.assertEqual(u.dictmerge(Da, D_tuple), D_merged)
        # Merge a dictionary with a list of dictionaries
        self.assertEqual(u.dictmerge(Da, D_list), D_merged)
        
    def test_intify(self):
        """ Testing the None-insensitive integer conversion function.
        """
        
        # Convert a list containing None
        self.assertEqual(u.intify(*[0.7, None, 6]), [0, None, 6])
        # Convert a tuple containing None
        self.assertEqual(u.intify(*(0.7, None, 6)), [0, None, 6])
        # Convert a 1D numpy array containing None
        self.assertEqual(u.intify(*np.array([0.7, None, 6])), [0, None, 6])
        
    def test_to_odd(self):
        """ Testing the function for determining the nearest odd number.
        """
        
        self.assertEqual(list(map(u.to_odd, [3.3, 3.6, 4.1, 4.6])),
                         [3, 3, 5, 5])
        
    def test_replist(self):
        """ Test 2D list repetition function.
        """
        
        self.assertEqual(u.replist(True, 2, 3),
                         [[True, True, True], [True, True, True]])
    

if __name__ == '__main__':
    unit.main()