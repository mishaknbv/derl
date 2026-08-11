""" Helper functions for learner tests. """
import numpy as np
import torch
from derl.testing import assert_all_close


def save_interactions(fname, alg):
  """ Saves interactions to a file. """
  interactions = next(alg.runner.run())
  np.savez(fname, **interactions)


def assert_interactions(fname, alg, ignore_keys=("state", "infos"),
                        rtol=1e-7, atol=0.):
  """ Checks that interactions have values from the file. """
  ignore_keys = set(ignore_keys) or {}
  interactions = next(alg.runner.run())
  with np.load(fname, allow_pickle=True) as expected:
    assert sorted(list(interactions.keys())) == sorted(list(expected.keys()))
    for key in filter(lambda k: k not in ignore_keys, expected.keys()):
      assert_all_close(interactions[key], expected[key], rtol=rtol, atol=atol)


def save_grad(fname, alg):
  """ Saves gradient to the file. """
  interactions = next(alg.runner.run())
  loss = alg.loss(interactions)
  loss.backward()
  grads = {f"grad_{i}": param.grad for i, param in
           enumerate(alg.model.parameters())}
  torch.save(grads, fname)


def assert_grad(fname, alg, rtol=1e-7, atol=0.):
  """ Checks that the gradients are close to the values from the file. """
  interactions = next(alg.runner.run())
  loss = alg.loss(interactions)
  loss.backward()
  expected = torch.load(fname)
  for i, param in enumerate(alg.model.parameters()):
    assert_all_close(param.grad, expected[f"grad_{i}"], rtol=rtol, atol=atol)


def save_losses(filename, num_losses, alg):
  """ Saves losses to the file. """
  data_iter = alg.runner.run()
  losses = []
  for _ in range(num_losses):
    losses.append(alg.step(next(data_iter)))
  torch.save(torch.tensor(losses), filename)


def assert_losses(filename, alg, rtol=1e-6, atol=0.):
  """ Checks that loss values are close to those from the file. """
  expected = torch.load(filename)
  data_iter = alg.runner.run()
  for i in range(expected.shape[0]):
    loss = alg.step(next(data_iter))
    assert_all_close(loss, expected[i], rtol=rtol, atol=atol)
