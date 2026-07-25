# pylint: disable=missing-docstring
from derl.env.make_env import make as make_env
from derl.factory.factory import Config
from derl.factory.dqn import DQNFactory
from derl.alg.test import AlgTestCase


class DQNTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    config = Config.make_for_factory(DQNFactory, "atari", args=[])
    config.storage_init_size = 42
    del config.num_recordings
    self.env = make_env("SpaceInvadersNoFrameskip-v4", nenvs=None, seed=0)
    self.alg = DQNFactory(config).make(self.env)
    self.alg.model.to("cpu")

  def test_interactions(self):
    _ = next(self.alg.runner.run())
