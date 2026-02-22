import unittest
import numpy as np
import scipy as sp
import coordinate

class TestPeriodicDist(unittest.TestCase):

    def setUp(self):
        """Define test data used across multiple tests."""
        self.domain = (0, 0, 1, 1)
        self.pos = np.array([
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.52012024, 0.9977939], 
            [0.49485442, 0.01256627],
            [0.03131782, 0.49644336], 
            [0.99977804, 0.51482676]
        ])

    def test_euclidean_consistency(self):
        """Test that with periodic=False, it matches standard scipy cdist."""
        from scipy.spatial.distance import cdist
        
        result = coordinate.periodic_dist(self.pos, periodic=False)
        expected = cdist(self.pos, self.pos)
        
        # Check if arrays are almost equal (handling floating point precision)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_periodic_wrap_around(self):
        """Test that distances across the boundary are correctly minimized."""
        # Points very close to the top and bottom edges
        top_point = np.array([[0.5, 0.99]])
        bottom_point = np.array([[0.5, 0.01]])
        test_points = np.vstack([top_point, bottom_point])
        
        # In a 1.0 unit box, the periodic distance should be 0.02 (0.99 to 1.0 to 0.01)
        # instead of the Euclidean 0.98.
        dists = coordinate.periodic_dist(test_points, domain=self.domain, periodic=True)
        
        # dists[0, 1] is the distance between top and bottom
        self.assertAlmostEqual(dists[0, 1], 0.02, places=7)

    def test_max_distance_limit(self):
        """In a 1x1 periodic box, no two points should be further than sqrt(0.5)."""
        rands = np.random.uniform(0, 1, (100, 2))
        dists = coordinate.periodic_dist(rands, domain=self.domain, periodic=True)
        
        max_dist = np.max(dists)
        theoretical_max = np.sqrt(0.5) # Distance to the center of a unit square
        
        self.assertLessEqual(max_dist, theoretical_max + 1e-7)

    def test_identity(self):
        """The distance from a point to itself should always be zero."""
        dists = coordinate.periodic_dist(self.pos, periodic=True)
        for i in range(len(self.pos)):
            self.assertEqual(dists[i, i], 0.0)

if __name__ == '__main__':
    unittest.main()