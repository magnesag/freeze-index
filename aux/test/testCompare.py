"""!
    Compare Module Unittests
    ========================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import unittest as ut

import numpy as np

from .. import compare


class TestCompareFunctions(ut.TestCase):
    def test_standardize(self):
        x = np.array([1, 2, 3, 4, 5])
        result = compare.standardize(x)
        self.assertAlmostEqual(np.mean(result), 0, places=7)
        self.assertAlmostEqual(np.std(result), 1, places=7)

    def test_resample_to_n_samples(self):
        x = np.array([1, 2, 3, 4, 5])

        # Test when n is equal to original length
        result = compare.resample_to_n_samples(x, 5)
        np.testing.assert_array_equal(result, x)

        # Test when n is greater than original length
        result = compare.resample_to_n_samples(x, 10)
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(result[-1], 5)

        # Test when n is less than original length
        result = compare.resample_to_n_samples(x, 3)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(result[-1], 5)

    def test_compare_signals(self):
        xs = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        names = ["A", "B", "C"]
        result = compare.compare_signals(xs, names)

        self.assertIsInstance(result, compare.ComparisonMetrics)
        self.assertEqual(result.names, names)

        # Check shapes
        self.assertEqual(result.mad.shape, (3, 3))
        self.assertEqual(result.rho.shape, (3, 3))
        self.assertEqual(result.r2.shape, (3, 3))

        # Check diagonal of r2
        np.testing.assert_array_almost_equal(np.diag(result.r2), np.ones(3))

        # Check symmetry of matrices
        np.testing.assert_array_almost_equal(result.mad, result.mad.T)
        np.testing.assert_array_almost_equal(result.rho, result.rho.T)
        np.testing.assert_array_almost_equal(result.r2, result.r2.T)

        # Check values
        np.testing.assert_array_almost_equal(
            result.mad, np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        )
        np.testing.assert_array_almost_equal(result.rho, np.ones((3, 3)))
        np.testing.assert_array_almost_equal(
            result.r2, np.array([[1, -0.5, -5], [-0.5, 1, -0.5], [-5, -0.5, 1]])
        )
