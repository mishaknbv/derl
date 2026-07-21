# pylint: disable=missing-docstring
from functools import partial
import torch
from derl.alg.test import AlgTestCase
from derl.env.make_env import make as make_env
from derl.factory.ppo import PPOFactory


class PPOAtariTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    kwargs = PPOFactory.get_kwargs()
    self.env = make_env("BreakoutNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.alg = PPOFactory(**kwargs).make(self.env)
    self.alg.model.load_state_dict(
        torch.load("testdata/ppo/atari/model.pt"))
    self.alg.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/ppo/atari/interactions.npz",
                             rtol=1e-6, atol=1e-6)

  def test_grad(self):
    self.assert_grad("testdata/ppo/atari/grads.npz", rtol=1e-6, atol=1e-6)

  def test_losses(self):
    self.assert_losses("testdata/ppo/atari/losses.npy", rtol=1e-5, atol=1e-5)


class PPOMuJoCoTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    kwargs = PPOFactory.get_kwargs("mujoco")
    # Modify some hyper parameters in order for the test not to take to long
    kwargs["num_runner_steps"] = 12
    kwargs["num_minibatches"] = 2
    kwargs["num_epochs"] = 3
    self.env = make_env("HalfCheetah-v5",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.env.reset = partial(self.env.reset, seed=0)
    self.alg = PPOFactory(**kwargs).make(self.env)
    self.alg.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/ppo/mujoco/interactions.npz",
                             rtol=0, atol=1e-4)

  def test_grad(self):
    self.assert_grad("testdata/ppo/mujoco/grads.npz", rtol=1e-5, atol=1e-5)

  def test_losses(self):
    self.assert_losses("testdata/ppo/mujoco/losses.npy", rtol=1e-5, atol=1e-5)
