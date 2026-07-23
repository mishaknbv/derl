""" Defies test case for testing models written in pytorch. """
import random
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import torch
import torch.testing as tt


class TorchTestCase(TestCase):
  """ Test case for testing code with models written in pytorch. """
  def setUp(self):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

  # pylint: disable=invalid-name
  def assertAllClose(self, actual, expected, rtol=1e-7, atol=0.):
    """ Checks that actual and expected arrays or torch tensors are equal. """
    self.assertEqual(type(actual), type(expected))
    if isinstance(actual, np.ndarray):
      nt.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, torch.Tensor):
      tt.assert_close(actual, expected, rtol=rtol, atol=atol)
    else:
      raise TypeError(f"unsupported type {type(actual)=}")
