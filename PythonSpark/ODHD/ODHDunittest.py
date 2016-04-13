'''
Created on Apr 9, 2016

@author: Fatemeh
'''
import unittest
from ParallelGA import crossover


class Test(unittest.TestCase):


    def test_crossoverTest(self):
        ch1, ch2 = crossover([0,3], [3,4])
        self.assertEqual((ch1 , ch2) , ([0,3], [3,4]))


if __name__ == "__main__":
    unittest.main()