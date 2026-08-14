# pylint: disable=missing-docstring
import numpy as np
import torch

from derl.models import LSTMModel, MuJoCoModel, NatureCNNModel
from derl.policies import ActorCriticPolicy, EpsilonGreedyPolicy
from derl.testing import assert_all_close


def test_actor_critic_categorical():
    model = NatureCNNModel((6, 1))
    model.to("cpu")
    policy = ActorCriticPolicy(model)
    act = policy.act(torch.rand(84, 84, 4))
    assert list(act.keys()) == ["actions", "log_prob", "values"]
    assert act["actions"] == np.array(1)
    assert_all_close(act["log_prob"], np.array(-1.721119), rtol=1e-6)
    assert_all_close(act["values"], np.array([0.257305294]), rtol=1e-5)


def test_actor_critic_recurrent():
    model = LSTMModel.make_cnn((84, 84, 1), (6, 1))
    policy = ActorCriticPolicy(model)
    state = policy.get_initial_state(1)
    act = policy.act(torch.randn(84, 84, 1), state, torch.zeros(1, dtype=bool))
    assert list(act.keys()) == ["actions", "log_prob", "values", "policy_state"]
    assert act["actions"] == np.array(5)
    assert_all_close(act["log_prob"], np.array(-1.7282835), rtol=1e-6)
    assert_all_close(act["values"], np.array(0.09740922), rtol=1e-6)
    assert act["policy_state"].shape ==  (1, 512)


def test_actor_critic_normal():
    model = MuJoCoModel(3, (2, 1))
    model.to("cpu")
    policy = ActorCriticPolicy(model)
    act = policy.act(torch.randn(3))
    assert list(act.keys()) == ["actions", "log_prob", "values"]
    assert_all_close(act["actions"], np.array([-1.7938228, 1.0464325]), rtol=1e-6)
    assert_all_close(act["log_prob"], np.array(-3.7467263))
    assert_all_close(act["values"], np.array([-0.18482158]), rtol=1e-6)


def assert_act(policy, expected):
    act = policy.act(torch.randn(84, 84, 4))
    assert list(act.keys()) == ["actions"]
    assert act["actions"] == expected


def test_epsilon_greedy_quantile_model():
    model = NatureCNNModel(8, num_quantiles=10)
    policy = EpsilonGreedyPolicy.quantile(model, epsilon=0)
    assert_act(policy, np.array(5))


def test_epsilon_greedy_dqn():
    model = NatureCNNModel(12)
    policy = EpsilonGreedyPolicy(model)
    assert_act(policy, np.array(2))
