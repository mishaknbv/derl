# pylint: disable=missing-docstring, redefined-outer-name
from functools import partial

import torch
import pytest

from derl.alg.sac import SACLossTuple
from derl.alg.test import assert_interactions
from derl.env.make_env import make as make_env
from derl.factory import Config
from derl.factory.sac import SACFactory
from derl.testing import assert_all_close


def iter_sac_loss_tuple(loss):
  """ Yields losses in SACLoss and calls backward on each. """
  if not isinstance(loss, SACLossTuple):
    raise TypeError("loss is expected to be of type SACLossTuple, "
                    f"got type(loss)={type(loss)}")
  yield "policy_loss", loss.policy_loss
  yield "entropy_scale_loss", loss.entropy_scale_loss
  for i, qvloss in enumerate(loss.qvalue_losses):
    yield f"qvalue_losses_{i}", qvloss


def save_grad(fname, alg):
  interactions = next(alg.runner.run())
  loss = alg.loss(interactions)
  grads = {}
  for field, lss in iter_sac_loss_tuple(loss):
    lss.backward()
    new_grads = {
        f"{field}/grad_{i}": param.grad
        for i, param in enumerate(alg.model.parameters())
    }
    alg.model.zero_grad()
    if set(grads) & set(new_grads):
      raise ValueError("intersection of gradient keys: "
                       f"{set(grads) & set(new_grads)}")
    grads.update(new_grads)
  torch.save(grads, fname)


def assert_grad(fname, alg, rtol=1e-7, atol=0.):
  interactions = next(alg.runner.run())
  loss = alg.loss(interactions)
  expected = torch.load(fname)
  for field, lss in iter_sac_loss_tuple(loss):
    lss.backward()
    for i, param in enumerate(alg.model.parameters()):
      if param.grad is None:
        assert param.grad == expected[f"{field}/grad_{i}"]
      else:
        assert_all_close(param.grad, expected[f"{field}/grad_{i}"],
                         rtol=rtol, atol=atol)
    alg.model.zero_grad()


def save_losses(filename, num_losses, alg):
  data_iter = alg.runner.run()
  losses = []
  for _ in range(num_losses):
    new_losses = []
    for _, lss in iter_sac_loss_tuple(alg.step(next(data_iter))):
      new_losses.append(lss.detach().item())
    losses.append(new_losses)
  torch.save(torch.tensor(losses), filename)


def assert_losses(filename, alg, rtol=1e-6, atol=0.):
  expected = torch.load(filename)
  data_iter = alg.runner.run()
  for i in range(expected.shape[0]):
    loss = alg.step(next(data_iter))
    for j, (_, lss) in enumerate(iter_sac_loss_tuple(loss)):
      assert_all_close(lss, expected[i][j], rtol=rtol, atol=atol)


@pytest.fixture
def sac_mujoco_alg():
  config = Config.make_for_factory(SACFactory, "mujoco", args=[])
  config.storage_size = 100
  config.storage_init_size = 10
  config.batch_size = 4
  config.steps_per_sample = 5
  config.num_storage_samples = 2
  del config.num_recordings
  env = make_env("HalfCheetah-v5", seed=0,
                 normalize_obs=False, normalize_ret=False)
  env.reset = partial(env.reset, seed=0)
  alg = SACFactory(config).make(env)
  alg.model.to("cpu")
  alg.loss_fn.target_policy.model.to("cpu")
  return alg


def test_sac_interactions(sac_mujoco_alg):
  assert_interactions("testdata/sac/mujoco/interactions.npz",
                      sac_mujoco_alg, rtol=1e-5, atol=1e-4)


def test_sac_grad(sac_mujoco_alg):
  assert_grad("testdata/sac/mujoco/grads.pt", sac_mujoco_alg,
              rtol=1e-4, atol=1e-4)


def test_sac_losses(sac_mujoco_alg):
  assert_losses("testdata/sac/mujoco/losses.pt", sac_mujoco_alg,
                rtol=1e-5, atol=1e-5)
