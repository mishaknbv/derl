"""Pytest configuration and shared fixtures for derl tests."""

import random

import numpy as np
import pytest
import torch

from derl import summary


@pytest.fixture(autouse=True)
def seed_random():
    """Seeds all randomness sources and enables deterministic algorithms."""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@pytest.fixture(autouse=True)
def disable_summaries(monkeypatch):
    """ Disables summary recording during tests. """
    summary.stop_recording()
    monkeypatch.setattr(summary, "should_record", lambda *args, **kwargs: False)
