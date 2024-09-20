"""!
    Compare Module Unittests
    ========================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import unittest as ut

import numpy as np

from aux import compare


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


class TestComparisonMetrics(ut.TestCase):
    def setUp(self):
        self.names = ["a", "b", "c", "d"]
        self.mad = np.array(
            [
                [0.0, 1.5, 1.0, 0.5],
                [1.5, 0.0, 0.5, 0.25],
                [1.0, 0.5, 0.0, 0.12],
                [0.5, 0.25, 0.12, 0.0],
            ]
        )
        self.rho = np.array(
            [
                [1.0, 0.3, 0.5, 0.7],
                [0.3, 1.0, 0.6, 0.8],
                [0.5, 0.6, 1.0, 0.9],
                [0.7, 0.8, 0.9, 1.0],
            ]
        )
        self.r2 = np.array(
            [
                [1.0, -0.2, 0.2, 0.7],
                [-0.2, 1.0, 0.5, 0.8],
                [0.2, 0.5, 1.0, 0.9],
                [0.7, 0.8, 0.9, 1.0],
            ]
        )
        self.metrics = compare.ComparisonMetrics(
            self.mad, self.rho, self.r2, self.names
        )

    def test_iterator(self):
        cr = [x for x in self.metrics]
        np.testing.assert_array_almost_equal(cr[0], self.mad)
        np.testing.assert_array_almost_equal(cr[1], self.rho)
        np.testing.assert_array_almost_equal(cr[2], self.r2)

    def test_compute_metrics_iou(self):
        cr = self.metrics.compute_metrics_iou("a")
        self.assertEqual(3, len(cr))
        self.assertAlmostEqual(0.0, cr[0])
        self.assertAlmostEqual(1.0 / 6.0, cr[1])
        self.assertAlmostEqual(2.0 / 11.0, cr[2])

    def test_compute_metrics_iou_error(self):
        self.assertRaises(ValueError, self.metrics.compute_metrics_iou, "jimmy")

    def test_compute_iou_match(self):
        a = [0.0, 1.0, 0.5]
        cr = compare.ComparisonMetrics.compute_iou(a, a)
        self.assertAlmostEqual(1.0, cr)

    def test_compute_iou_inclusion(self):
        a = [0.0, 1.0, 0.5]
        b = [0.25, 0.75]
        cr = compare.ComparisonMetrics.compute_iou(a, b)
        self.assertAlmostEqual(0.5, cr)

    def test_compute_iou_overlap(self):
        a = [0.0, 0.75]
        b = [0.5, 1.0]
        cr = compare.ComparisonMetrics.compute_iou(a, b)
        self.assertAlmostEqual(0.25, cr)

    def test_compute_iou_no_intersection(self):
        a = [0, 0.5]
        b = [0.75, 1.0]
        cr = compare.ComparisonMetrics.compute_iou(a, b)
        self.assertAlmostEqual(0.0, cr)
