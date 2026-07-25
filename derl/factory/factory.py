""" Code to construct different objects. """
from abc import ABC, abstractmethod
import argparse
from math import floor


class Config:
  """ Algorithm configuration dictionary. """
  def __init__(self, **config):
    self.unused = set()
    for key, val in config.items():
      key = key.replace('-', '_')
      setattr(self, key, val)
      self.unused.add(key)

  @classmethod
  def make_for_factory(cls, factory_class, args_type="atari", args=None):
    """ Creates a config for a factory. """
    argdict = factory_class.get_argdict(args_type)
    parser = argparse.ArgumentParser()
    for key, val in argdict.items():
      if isinstance(val, dict):
        parser.add_argument(f"--{key}", **val)
      else:
        parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args(args)
    return cls(**vars(args))

  def __getattribute__(self, name):
    unused = super().__getattribute__("unused")
    unused.discard(name)
    return super().__getattribute__(name)

  def __delattr__(self, name):
    self.unused.discard(name)
    super().__delattr__(name)

  def __ior__(self, other: dict):
    for key, val in other.items():
      key = key.replace('-', '_')
      setattr(self, key, val)
    return self


class Factory(ABC):
  """ Factory to construct learning algorithms. """
  def __init__(self, config):
    self.config = config

  @staticmethod
  @abstractmethod
  def get_argdict(args_type="atari"):
    """ Returns default argument dictionary for argument parsing. """

  def __getattr__(self, name):
    return getattr(self.config, name)

  def make_env_kwargs(self, env_id):
    """ Returns keyword arguments for derl.env.make function. """
    _ = env_id
    recording_period = floor(
      self.config.num_train_steps // self.config.num_recordings)
    return dict(recording_period=recording_period,
                nenvs=getattr(self.config, "nenvs", None))

  @abstractmethod
  def make_runner(self, env, model=None, nlogs=1e5, **kwargs):
    """ Creates and returns algorithm runner. """

  @abstractmethod
  def make_trainer(self, runner, **kwargs):
    """ Creates and returns algorithm trainer. """

  @abstractmethod
  def make_alg(self, runner, trainer, **kwargs):
    """ Creates and returns alg instance with specified runner and trainer. """

  def make(self, env, nlogs=1e5, check_kwargs=True):
    """ Creates and returns algorithm instance. """
    runner = self.make_runner(env, nlogs=nlogs)
    trainer = self.make_trainer(runner)
    alg = self.make_alg(runner, trainer)
    if check_kwargs and self.config.unused:
      raise ValueError(
          "constructing target object does not use all keyword arguments, "
          f"unused keyword arguments are: {self.config.unused}; ")
    return alg
