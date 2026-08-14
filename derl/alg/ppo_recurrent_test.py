# pylint: disable=missing-docstring, redefined-outer-name
from functools import partial

import pytest
import torch

from derl.alg.test import assert_grad, assert_interactions, assert_losses
from derl.env.make_env import make as make_env
from derl.factory import Config
from derl.factory.ppo import PPOFactory
from derl.models import LSTMModel


@pytest.fixture
def ppo_recurrent_atari_alg():
    config = Config.make_for_factory(PPOFactory, args=["--recurrent"])
    config.num_epochs = 2
    config.num_minibatches = 4
    del config.num_recordings
    env = make_env("BreakoutNoFrameskip-v4", nenvs=config.nenvs, seed=0,
                   num_queue_frames=1)
    alg = PPOFactory(config).make(env)
    assert isinstance(alg.model, LSTMModel)
    assert alg.runner.policy.is_recurrent()
    alg.model.load_state_dict(torch.load("testdata/ppo/recurrent/model.pt"))
    alg.model.to("cpu")
    return alg


def test_ppo_recurrent_atari_interactions(ppo_recurrent_atari_alg):
    assert_interactions(
        "testdata/ppo/recurrent/interactions.npz",
        ppo_recurrent_atari_alg,
        rtol=1e-6,
        atol=1e-6,
    )


def test_ppo_recurrent_atari_grad(ppo_recurrent_atari_alg):
    assert_grad(
        "testdata/ppo/recurrent/grads.pt", ppo_recurrent_atari_alg, rtol=1e-6, atol=1e-6
    )


def test_ppo_recurrent_atari_losses(ppo_recurrent_atari_alg):
    runner = ppo_recurrent_atari_alg.runner.runner
    assert runner.num_epochs == 2
    assert runner.num_minibatches == 4
    assert not runner.shuffle_before_epoch
    assert_losses(
        "testdata/ppo/recurrent/losses.pt", ppo_recurrent_atari_alg, rtol=1e-5, atol=1e-5
    )


@pytest.fixture
def ppo_recurrent_mujoco_alg():
    config = Config.make_for_factory(PPOFactory, "mujoco", args=["--recurrent"])
    config.num_runner_steps = 12
    config.num_minibatches = 2
    config.num_epochs = 3
    del config.num_recordings
    env = make_env("HalfCheetah-v5", nenvs=config.nenvs, seed=0,
                   normalize_obs=False, normalize_ret=False)
    env.reset = partial(env.reset, seed=0)
    alg = PPOFactory(config).make(env)
    assert isinstance(alg.model, LSTMModel)
    assert alg.runner.policy.is_recurrent()
    alg.model.load_state_dict(torch.load("testdata/ppo/recurrent/mujoco/model.pt"))
    alg.model.to("cpu")
    return alg


def test_ppo_recurrent_mujoco_interactions(ppo_recurrent_mujoco_alg):
    assert_interactions(
        "testdata/ppo/recurrent/mujoco/interactions.npz",
        ppo_recurrent_mujoco_alg,
        rtol=1e-6,
        atol=1e-6,
    )


def test_ppo_recurrent_mujoco_grad(ppo_recurrent_mujoco_alg):
    assert_grad(
        "testdata/ppo/recurrent/mujoco/grads.pt",
        ppo_recurrent_mujoco_alg,
        rtol=1e-6,
        atol=1e-6,
    )


def test_ppo_recurrent_mujoco_losses(ppo_recurrent_mujoco_alg):
    runner = ppo_recurrent_mujoco_alg.runner.runner
    assert runner.horizon == 12
    assert runner.num_epochs == 3
    assert runner.num_minibatches == 2
    assert not runner.shuffle_before_epoch
    assert_losses(
        "testdata/ppo/recurrent/mujoco/losses.pt",
        ppo_recurrent_mujoco_alg,
        rtol=1e-5,
        atol=1e-5,
    )
