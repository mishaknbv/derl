# pylint: disable=missing-docstring, redefined-outer-name
import collections

import numpy.testing as nt
import pytest
import torch

from derl.models import MuJoCoModel, NatureCNNBase, NatureCNNModel, NoisyLinear
from derl.testing import assert_all_close, assert_orthogonal


def test_noisy_linear_parameters():
    layer = NoisyLinear(3, 4)
    assert len(list(layer.parameters())) == 4


def test_noisy_linear_call():
    layer = NoisyLinear(3, 4)
    assert layer(torch.randn(2, 3)).shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cuda")
def test_noisy_linear_cuda_call():
    layer = NoisyLinear(3, 4)
    layer.to("cuda")
    assert layer(torch.randn(2, 3).to("cuda")).shape == (2, 4)


@pytest.fixture
def dqn_base():
    dqn_base = NatureCNNBase()
    dqn_base.load_state_dict(torch.load("testdata/models/dqn-base.pt"))
    return dqn_base


def test_dqn_base_params(dqn_base):
    assert len(list(dqn_base.parameters())) == 8


def test_dqn_base_noisy_params():
    base = NatureCNNBase(noisy=True)
    assert len(list(base.parameters())) == 10


def test_dqn_base_call(dqn_base):
    inputs = torch.rand(32, 84, 84, 4)
    outputs = dqn_base(inputs)
    expected = torch.load("testdata/models/dqn-base-outputs.pt")
    assert_all_close(outputs, expected, atol=1e-6)


@pytest.fixture
def nature_cnn_actor_critic():
    dqn = NatureCNNModel(output_units=(4, 1))
    dqn.to("cpu")
    return dqn


def test_nature_cnn_params(nature_cnn_actor_critic):
    dqn = nature_cnn_actor_critic
    nweights = nbiases = 0
    for module in dqn.modules():
        if hasattr(module, "bias"):
            nt.assert_equal(module.bias.detach().numpy(), 0.0)
            nbiases += 1
        if hasattr(module, "weight"):
            assert_orthogonal(module.weight.detach().numpy())
            nweights += 1
    assert nweights == 6
    assert nbiases == 6


def test_nature_cnn_broadcast():
    dqn = NatureCNNModel(output_units=(4, 1))
    inputs = torch.rand(84, 84, 4)
    outputs = dqn(inputs)
    assert len(outputs) == 2
    assert outputs[0].shape == torch.Size((4,))
    assert outputs[1].shape == torch.Size((1,))


def test_distributional_output_shape():
    dqn = NatureCNNModel(output_units=6, num_quantiles=200)
    outputs = dqn(torch.rand(32, 84, 84, 4))
    assert outputs.shape == torch.Size([32, 6, 200])


def test_mujoco_model_params():
    model = MuJoCoModel(4, 5)
    model.to("cpu")
    nweights = nbiases = 0
    for module in model.modules():
        if hasattr(module, "bias"):
            nt.assert_equal(module.bias.detach().numpy(), 0.0)
            nbiases += 1
        if hasattr(module, "weight"):
            assert_orthogonal(module.weight.detach().numpy())
            nweights += 1
    assert nweights == 3
    assert nbiases == 3

    # The model should also contain 1 logstd parameter
    assert len(list(model.parameters())) == 6 + 1


def test_mujoco_model_call():
    model = MuJoCoModel(4, (5, 1))
    outputs = model(torch.rand(2, 4))
    assert isinstance(outputs, collections.abc.Iterable)
    assert len(outputs) == 3

    mean, std, values = outputs
    assert mean.shape == (2, 5)
    assert std.shape == (2, 5)
    nt.assert_equal(std.cpu().detach().numpy(), 1.0)
    assert values.shape == (2, 1)


def test_mujoco_model_broadcast():
    model = MuJoCoModel(3, 5)
    outputs = model(torch.rand(3))
    assert isinstance(outputs, collections.abc.Iterable)
    assert len(outputs) == 2
    assert outputs[0].shape == (5,)
    assert outputs[1].shape == (5,)


def test_mujoco_model_dtype():
    model = MuJoCoModel(3, 5)
    outputs = model(torch.rand(3).double())
    assert outputs[0].shape == (5,)
    assert outputs[1].shape == (5,)
