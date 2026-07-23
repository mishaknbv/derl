# pylint: disable=missing-docstring
from functools import partial
import torch
from derl.alg.test import AlgTestCase
from derl.alg.sac import SACLossTuple
from derl.env.make_env import make as make_env
from derl.factory.sac import SACFactory


def iter_sac_loss_tuple(loss):
  """ Yields losses in SACLoss and calls backward on each. """
  if not isinstance(loss, SACLossTuple):
    raise TypeError("loss is expected to be of type SACLossTuple, "
                    f"got type(loss)={type(loss)}")
  yield "policy_loss", loss.policy_loss
  yield "entropy_scale_loss", loss.entropy_scale_loss
  for i, qvloss in enumerate(loss.qvalue_losses):
    yield f"qvalue_losses_{i}", qvloss


class SACMuJoCoTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    kwargs = SACFactory.get_kwargs("mujoco")
    kwargs["storage_size"] = 100
    kwargs["storage_init_size"] = 10
    kwargs["batch_size"] = 4
    kwargs["steps_per_sample"] = 5
    kwargs["num_storage_samples"] = 2
    self.env = make_env("HalfCheetah-v5", seed=0,
                        normalize_obs=False, normalize_ret=False)
    self.env.reset = partial(self.env.reset, seed=0)
    self.alg = SACFactory(**kwargs).make(self.env)
    self.alg.model.to("cpu")
    self.alg.loss_fn.target_policy.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/sac/mujoco/interactions.npz",
                             rtol=1e-5, atol=1e-4)

  def save_grad(self, fname):
    interactions = next(self.alg.runner.run())
    loss = self.alg.loss(interactions)
    grads = {}
    for field, lss in iter_sac_loss_tuple(loss):
      lss.backward()
      new_grads = {
          f"{field}/grad_{i}": param.grad
          for i, param in enumerate(self.alg.model.parameters())
      }
      self.alg.model.zero_grad()
      if set(grads) & set(new_grads):
        raise ValueError("intersection of gradient keys: "
                         f"{set(grads) & set(new_grads)}")
      grads.update(new_grads)
    torch.save(grads, fname)

  def assert_grad(self, fname, rtol=1e-7, atol=0.):
    interactions = next(self.alg.runner.run())
    loss = self.alg.loss(interactions)
    expected = torch.load(fname)
    for field, lss in iter_sac_loss_tuple(loss):
      lss.backward()
      for i, param in enumerate(self.alg.model.parameters()):
        with self.subTest(field=field, grad_i=i):
          if param.grad is None:
            self.assertEqual(param.grad, expected[f"{field}/grad_{i}"])
          else:
            self.assertAllClose(param.grad, expected[f"{field}/grad_{i}"],
                                rtol=rtol, atol=atol)
      self.alg.model.zero_grad()

  def test_grad(self):
    self.assert_grad("testdata/sac/mujoco/grads.pt", rtol=1e-4, atol=1e-4)

  def save_losses(self, filename, num_losses):
    data_iter = self.alg.runner.run()
    losses = []
    for _ in range(num_losses):
      new_losses = []
      for _, lss in iter_sac_loss_tuple(self.alg.step(next(data_iter))):
        new_losses.append(lss.detach().item())
      losses.append(new_losses)
    torch.save(torch.tensor(losses), filename)

  def assert_losses(self, filename, rtol=1e-6, atol=0.):
    expected = torch.load(filename)
    data_iter = self.alg.runner.run()
    for i in range(expected.shape[0]):
      loss = self.alg.step(next(data_iter))
      for j, (name, lss) in enumerate(iter_sac_loss_tuple(loss)):
        with self.subTest(i=i, name=name):
          self.assertAllClose(lss, expected[i][j], rtol=rtol, atol=atol)

  def test_losses(self):
    self.assert_losses("testdata/sac/mujoco/losses.pt", rtol=1e-5, atol=1e-5)
