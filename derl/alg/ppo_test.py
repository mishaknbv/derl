# pylint: disable=missing-docstring, redefined-outer-name
from functools import partial

import torch
import pytest

from derl.alg.test import assert_grad, assert_interactions, assert_losses
from derl.env.make_env import make as make_env
from derl.factory import Config
from derl.factory.ppo import PPOFactory


@pytest.fixture
def ppo_atari_alg():
  config = Config.make_for_factory(PPOFactory, args=[])
  config.num_epochs = 2
  config.num_minibatches = 3
  del config.num_recordings
  env = make_env("BreakoutNoFrameskip-v4", nenvs=config.nenvs, seed=0)
  alg = PPOFactory(config).make(env)
  alg.model.load_state_dict(torch.load("testdata/ppo/atari/model.pt"))
  alg.model.to("cpu")
  return alg


def test_ppo_atari_interactions(ppo_atari_alg):
  assert_interactions("testdata/ppo/atari/interactions.npz",
                      ppo_atari_alg, rtol=1e-6, atol=1e-6)


def test_ppo_atari_grad(ppo_atari_alg):
  assert_grad("testdata/ppo/atari/grads.pt", ppo_atari_alg, rtol=1e-6, atol=1e-6)


def test_ppo_atari_losses(ppo_atari_alg):
  assert ppo_atari_alg.runner.runner.num_epochs == 2
  assert ppo_atari_alg.runner.runner.num_minibatches == 3
  assert_losses("testdata/ppo/atari/losses.pt", ppo_atari_alg,
                rtol=1e-5, atol=1e-5)


@pytest.fixture
def ppo_mujoco_alg():
  config = Config.make_for_factory(PPOFactory, "mujoco", args=[])
  # Modify some hyper parameters in order for the test not to take too long
  config.num_runner_steps = 12
  config.num_minibatches = 2
  config.num_epochs = 3
  del config.num_recordings
  env = make_env("HalfCheetah-v5", nenvs=config.nenvs, seed=0)
  env.reset = partial(env.reset, seed=0)
  alg = PPOFactory(config).make(env)
  alg.model.to("cpu")
  return alg


def test_ppo_mujoco_interactions(ppo_mujoco_alg):
  assert_interactions("testdata/ppo/mujoco/interactions.npz",
                      ppo_mujoco_alg, rtol=0, atol=1e-4)


def test_ppo_mujoco_grad(ppo_mujoco_alg):
  assert_grad("testdata/ppo/mujoco/grads.pt", ppo_mujoco_alg,
              rtol=1e-5, atol=1e-5)


def test_ppo_mujoco_losses(ppo_mujoco_alg):
  assert ppo_mujoco_alg.runner.runner.num_epochs == 3
  assert ppo_mujoco_alg.runner.runner.num_minibatches == 2
  assert ppo_mujoco_alg.runner.horizon == 12
  assert_losses("testdata/ppo/mujoco/losses.pt", ppo_mujoco_alg,
                rtol=1e-5, atol=1e-5)
