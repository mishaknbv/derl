# pylint: disable=missing-docstring, redefined-outer-name
import gymnasium as gym
import numpy as np
import numpy.testing as nt
import pytest
import torch
from torch import nn

from derl.alg.rnd import (
    RNDGAE,
    IntrinsicReward,
    PredictorModel,
    RNDModel,
    reinitialize,
    replace_module,
)
from derl.alg.test import assert_grad, assert_interactions, assert_losses
from derl.env.make_env import make as make_env
from derl.factory import Config
from derl.factory.rnd import RNDFactory
from derl.models import collocate_inputs


class DummyEnv(gym.Env):
    """Minimal environment for testing the IntrinsicReward wrapper."""

    nenvs = 1

    def __init__(self):
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, 3, 1))
        self.action_space = gym.spaces.Discrete(2)

    def step(self, action):
        return np.ones((3, 3, 1), dtype=np.float32), 0.0, False, False, {}

    def reset(self, **_kwargs):
        return np.zeros((3, 3, 1), dtype=np.float32), {}

    def render(self):
        return None


class StubPolicy:
    """Policy stub returning fixed value predictions."""

    def __init__(self, values):
        self._values = values

    def act(self, obs, state=None, resets=None):
        del obs, state, resets
        return {"values": self._values}


def test_reinitialize():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    params_before = [param.detach().clone() for param in model.parameters()]
    nresets = reinitialize(model)
    assert nresets == 2
    for before, after in zip(params_before, model.parameters()):
        assert not torch.equal(before, after)


def test_reinitialize_init_fn():
    model = nn.Sequential(nn.Linear(4, 4))

    def init_fn(layer):
        if hasattr(layer, "weight"):
            nn.init.zeros_(layer.weight)

    nresets = reinitialize(model, init_fn)
    assert nresets == -1
    assert torch.all(model[0].weight == 0.0)


def test_replace_module():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    old = dict(model.named_children())
    replace_module(model, nn.Linear, lambda: nn.Linear(4, 4))
    new = dict(model.named_children())
    assert list(new) == list(old)
    for key, child in old.items():
        if isinstance(child, nn.Linear):
            assert new[key] is not child
        else:
            assert new[key] is child


def test_rnd_model_shapes():
    model = RNDModel(num_actions=4)
    model.to("cpu")
    policy, values = model(torch.rand(8, 84, 84, 4))
    assert policy.shape == (8, 4)
    assert values.shape == (8, 2)


def test_rnd_model_broadcast():
    model = RNDModel(num_actions=4)
    model.to("cpu")
    policy, values = model(torch.rand(84, 84, 4))
    assert policy.shape == (4,)
    assert values.shape == (2,)


def test_predictor_model_shapes():
    base = nn.Linear(4, 8)
    model = PredictorModel(base, torch.randn(2, 4), num_extra_layers=2)
    outputs = model(torch.randn(3, 4))
    assert outputs.shape == (3, 8)


class LinearModel(nn.Module):
    """Linear layer that collocates numpy inputs."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    @collocate_inputs(dtype=False)
    def forward(self, inputs):
        return self.linear(inputs)


def test_intrinsic_reward_wrapper():
    predictor = LinearModel()
    target = LinearModel()
    predictor.linear.weight.data = torch.tensor([[2.0], [3.0]])
    predictor.linear.bias.data = torch.tensor([0.5, -1.5])
    target.linear.weight.data = torch.tensor([[1.0], [1.0]])
    target.linear.bias.data = torch.tensor([0.0, 0.0])
    env = IntrinsicReward(
        DummyEnv(), predictor, target, normalize=False, intrinsic_gamma=0.99
    )
    env.reset()
    obs, rewards, terminated, truncated, info = env.step(0)
    assert not terminated
    assert not truncated
    assert info == {}

    extrinsic, intrinsic = rewards
    assert extrinsic == 0.0
    # obs[..., -1] == 1., predictor(1.) = [2.5, 1.5] and target(1.) = [1., 1.]
    # so the intrinsic reward is mean((2.5 - 1.)^2, (1.5 - 1.)^2) == 1.25
    nt.assert_allclose(obs, np.ones((3, 3, 1)))
    assert intrinsic.shape == (3, 3)
    nt.assert_allclose(intrinsic, 1.25, rtol=0, atol=1e-6)


def test_rnd_gae():
    # horizon=2, nenvs=1; the reward at each step is (extrinsic, intrinsic).
    # The episode terminates at the last step (extrinsic terminations only).
    trajectory = {
        "rewards": np.array([[[1.0], [3.0]], [[2.0], [4.0]]]),
        "terminations": np.array([[False], [True]]),
        "truncations": np.array([[False], [False]]),
        "values": np.array([[[0.5, 2.5]], [[1.5, 3.5]]]),
        "state": {"latest_observations": np.zeros((1, 84, 84, 4))},
    }
    policy = StubPolicy(np.array([[0.25, 0.75]]))
    gae = RNDGAE(
        policy,
        extrinsic_gamma=0.999,
        intrinsic_gamma=0.99,
        extrinsic_coef=2.0,
        intrinsic_coef=1.0,
        normalize=False,
    )
    advantages, value_targets = gae(trajectory)

    # Hand-computed GAE values for the trajectory above with
    # extrinsic_gamma=0.999, intrinsic_gamma=0.99 and lambda_=0.95.
    expected_advantages = np.array([[[2.473025, 5.13357125]], [[0.5, 1.2425]]])
    expected_value_targets = np.array([[[2.973025, 7.63357125]], [[2.0, 4.7425]]])
    nt.assert_allclose(advantages, expected_advantages, rtol=0, atol=1e-5)
    nt.assert_allclose(value_targets, expected_value_targets, rtol=0, atol=1e-5)
    nt.assert_allclose(
        trajectory["advantages"], np.array([[10.07962125], [2.2425]]), rtol=0, atol=1e-5
    )
    assert trajectory["rewards"].shape == (2, 1, 2)


@pytest.fixture
def rnd_atari_alg():
    config = Config.make_for_factory(RNDFactory, args=[])
    # Modify some hyper parameters in order for the test not to take too long
    config.num_train_steps = 64
    config.nenvs = 2
    config.num_runner_steps = 4
    config.num_epochs = 1
    config.num_minibatches = 2
    config.num_warmup_steps = 0
    del config.num_recordings
    env = make_env("BreakoutNoFrameskip-v4", nenvs=config.nenvs, seed=0)
    alg = RNDFactory(config).make(env)
    alg.model.to("cpu")
    alg.predictor.to("cpu")
    alg.target.to("cpu")
    return alg


def test_rnd_atari_interactions(rnd_atari_alg):
    assert rnd_atari_alg.runner.num_epochs == 1
    assert rnd_atari_alg.runner.num_minibatches == 2
    assert rnd_atari_alg.runner.unwrapped.horizon == 4
    assert_interactions(
        "testdata/rnd/atari/interactions.npz", rnd_atari_alg, rtol=1e-6, atol=1e-6
    )


def test_rnd_atari_grad(rnd_atari_alg):
    assert_grad("testdata/rnd/atari/grads.pt", rnd_atari_alg, rtol=1e-6, atol=1e-6)


def test_rnd_atari_losses(rnd_atari_alg):
    assert_losses("testdata/rnd/atari/losses.pt", rnd_atari_alg, rtol=1e-5, atol=1e-5)
