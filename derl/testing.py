"""Shared helpers for pytest tests."""

import numpy as np
import numpy.testing as nt
import torch
import torch.testing as tt


def assert_all_close(actual, expected, rtol=1e-7, atol=0.0):
    """Checks that actual and expected arrays or torch tensors are equal."""
    assert type(actual) is type(expected)
    if isinstance(actual, np.ndarray):
        nt.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, torch.Tensor):
        tt.assert_close(actual, expected, rtol=rtol, atol=atol)
    else:
        raise TypeError(f"unsupported type {type(actual)=}")


def assert_orthogonal(arr):
    """Checks that np.ndarray has orthogonal initialization."""
    arr = np.reshape(arr, (arr.shape[0], -1))
    nrows, ncols = arr.shape
    if nrows > ncols:
        nt.assert_allclose(arr.T @ arr, np.eye(ncols), atol=1e-5, rtol=1e-5)
    else:
        nt.assert_allclose(arr @ arr.T, np.eye(nrows), atol=1e-5, rtol=1e-5)
