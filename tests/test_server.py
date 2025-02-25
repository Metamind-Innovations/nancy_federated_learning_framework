import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.server.server import weighted_average


class TestServerFunctions(unittest.TestCase):
    
    def test_weighted_average(self):

        metrics = [0.9, 0.8, 0.5]
        weights = [100, 200, 50]
        expected_avg = (0.9 * 100 + 0.8 * 200 + 0.5 * 50) / 350
        
        result = weighted_average(metrics, weights)
        self.assertAlmostEqual(result, expected_avg)


if __name__ == '__main__':
    unittest.main()