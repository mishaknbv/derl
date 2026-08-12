# pylint: disable=missing-docstring
import numpy as np
import numpy.testing as npt

from derl.runners.sum_tree import SumTree


def test_sum_tree_three():
    sum_tree = SumTree(3)
    sum_tree.replace(np.asarray([0, 1, 2]), np.asarray([1.0, 2.0, 3.0]))
    assert sum_tree.sum == 6.0
    assert sum_tree.get_value(0) == 1.0
    assert sum_tree.get_value(1) == 2.0
    assert sum_tree.get_value(2) == 3.0
    actual = sum_tree.retrieve(np.asarray([0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 6.0, 6.5]))
    expected = np.asarray([0, 0, 1, 1, 1, 2, 2, -1])
    npt.assert_array_equal(actual, expected)
