# pylint: disable=missing-docstring, redefined-outer-name
import pytest

from derl.env.make_env import make as make_env
from derl.factory.factory import Config
from derl.factory.dqn import DQNFactory


@pytest.fixture
def dqn_alg():
  config = Config.make_for_factory(DQNFactory, "atari", args=[])
  config.storage_init_size = 42
  del config.num_recordings
  env = make_env("SpaceInvadersNoFrameskip-v4", nenvs=None, seed=0)
  alg = DQNFactory(config).make(env)
  alg.model.to("cpu")
  return alg


def test_dqn_interactions(dqn_alg):
  _ = next(dqn_alg.runner.run())
