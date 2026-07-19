# pylint: disable=missing-docstring
from unittest import TestCase
import numpy as np
import numpy.testing as npt
from derl.runners.sum_tree import SumTree


class SumTreeTest(TestCase):
  def test_three(self):
    sum_tree = SumTree(3)
    sum_tree.replace(np.asarray([0, 1, 2]), np.asarray([1., 2., 3.]))
    self.assertEqual(sum_tree.sum, 6.)
    self.assertEqual(sum_tree.get_value(0), 1.)
    self.assertEqual(sum_tree.get_value(1), 2.)
    self.assertEqual(sum_tree.get_value(2), 3.)
    actual = sum_tree.retrieve(np.asarray([0.5, 1., 1.5, 2.5, 3., 3.5, 6., 6.5]))
    expected = np.asarray([0, 0, 1, 1, 1, 2, 2, -1])
    npt.assert_array_equal(actual, expected)
