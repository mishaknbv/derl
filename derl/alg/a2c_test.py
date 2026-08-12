# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import torch

from derl.alg.test import assert_grad, assert_interactions, assert_losses
from derl.env.make_env import make as make_env
from derl.factory.a2c import A2CFactory
from derl.factory.factory import Config


@pytest.fixture
def a2c_alg():
    config = Config.make_for_factory(A2CFactory, args=[])
    del config.num_recordings
    env = make_env("SpaceInvadersNoFrameskip-v4", nenvs=config.nenvs, seed=0)
    alg = A2CFactory(config).make(env)
    alg.model.load_state_dict(torch.load("testdata/a2c/atari/model.pt"))
    alg.model.to("cpu")
    return alg


def test_a2c_interactions(a2c_alg):
    assert_interactions(
        "testdata/a2c/atari/interactions.npz", a2c_alg, rtol=1e-6, atol=1e-6
    )


def test_a2c_grad(a2c_alg):
    assert_grad("testdata/a2c/atari/grads.pt", a2c_alg, rtol=1e-6, atol=1e-6)


def test_a2c_losses(a2c_alg):
    assert_losses("testdata/a2c/atari/losses.pt", a2c_alg, rtol=1e-5, atol=1e-4)
