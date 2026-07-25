# pylint: disable=missing-docstring
from functools import partial
import torch
from derl.alg.test import AlgTestCase
from derl.env.make_env import make as make_env
from derl.factory import Config
from derl.factory.ppo import PPOFactory


class PPOAtariTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    config = Config.make_for_factory(PPOFactory, args=[])
    config.num_epochs = 2
    config.num_minibatches = 3
    del config.num_recordings
    self.env = make_env("BreakoutNoFrameskip-v4", nenvs=config.nenvs, seed=0)
    self.alg = PPOFactory(config).make(self.env)
    self.alg.model.load_state_dict(
        torch.load("testdata/ppo/atari/model.pt"))
    self.alg.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/ppo/atari/interactions.npz",
                             rtol=1e-6, atol=1e-6)

  def test_grad(self):
    self.assert_grad("testdata/ppo/atari/grads.pt", rtol=1e-6, atol=1e-6)

  def test_losses(self):
    self.assertEqual(self.alg.runner.runner.num_epochs, 2)
    self.assertEqual(self.alg.runner.runner.num_minibatches, 3)
    self.assert_losses("testdata/ppo/atari/losses.pt", rtol=1e-5, atol=1e-5)


class PPOMuJoCoTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    config = Config.make_for_factory(PPOFactory, "mujoco", args=[])
    # Modify some hyper parameters in order for the test not to take to long
    config.num_runner_steps = 12
    config.num_minibatches = 2
    config.num_epochs = 3
    del config.num_recordings
    self.env = make_env("HalfCheetah-v5", nenvs=config.nenvs, seed=0)
    self.env.reset = partial(self.env.reset, seed=0)
    self.alg = PPOFactory(config).make(self.env)
    self.alg.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/ppo/mujoco/interactions.npz",
                             rtol=0, atol=1e-4)

  def test_grad(self):
    self.assert_grad("testdata/ppo/mujoco/grads.pt", rtol=1e-5, atol=1e-5)

  def test_losses(self):
    self.assertEqual(self.alg.runner.runner.num_epochs, 3)
    self.assertEqual(self.alg.runner.runner.num_minibatches, 2)
    self.assertEqual(self.alg.runner.horizon, 12)
    self.assert_losses("testdata/ppo/mujoco/losses.pt", rtol=1e-5, atol=1e-5)
