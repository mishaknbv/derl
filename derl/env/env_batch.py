# pylint: skip-file
from multiprocessing import Process, Pipe

from gymnasium import Env, Wrapper, Space
import numpy as np


class SpaceBatch(Space):
  def __init__(self, spaces):
    first_type = type(spaces[0])
    first_shape = spaces[0].shape
    first_dtype = spaces[0].dtype
    for space in spaces:
      if not isinstance(space, first_type):
        raise TypeError("spaces have different types: {}, {}"
                        .format(first_type, type(space)))
      if first_shape != space.shape:
        raise ValueError("spaces have different shapes: {}, {}"
                         .format(first_shape, space.shape))
      if first_dtype != space.dtype:
        raise ValueError("spaces have different data types: {}, {}"
                         .format(first_dtype, space.dtype))

    self.spaces = spaces
    super(SpaceBatch, self).__init__(shape=self.spaces[0].shape,
                                     dtype=self.spaces[0].dtype)

  def sample(self):
    return np.stack([space.sample() for space in self.spaces])

  def __getattr__(self, attr):
    return getattr(self.spaces[0], attr)


class EnvBatch(Env):
  def __init__(self, make_env, nenvs=None):
    make_env_functions = self._get_make_env_functions(make_env, nenvs)
    self._envs = [make_env() for make_env in make_env_functions]
    self._nenvs = len(self.envs)
    self.action_space = SpaceBatch([env.action_space
                                    for env in self._envs])

  def _get_make_env_functions(self, make_env, nenvs):
    if nenvs is None and not isinstance(make_env, list):
      raise ValueError("When nenvs is None make_env"
                       " must be a list of callables")
    if nenvs is not None and not callable(make_env):
      raise ValueError("When nenvs is not None make_env must be callable")

    if nenvs is not None:
      make_env = [make_env for _ in range(nenvs)]
    return make_env

  @property
  def nenvs(self):
    return self._nenvs

  @property
  def envs(self):
    return self._envs

  def _check_actions(self, actions):
    if not len(actions) == self.nenvs:
      raise ValueError(
          "number of actions is not equal to number of envs: "
          "len(actions) = {}, nenvs = {}"
          .format(len(actions), self.nenvs))

  def step(self, actions):
    self._check_actions(actions)
    obs, rews, resets, infos = [], [], [], []
    for env, action in zip(self._envs, actions):
      ob, rew, done, info = env.step(action)
      if done:
        ob = env.reset()
      obs.append(ob)
      rews.append(rew)
      resets.append(done)
      infos.append(info)
    return np.stack(obs), np.stack(rews), np.stack(resets), infos

  def reset(self):
    obs, infos = list(zip(*[env.reset() for env in self.envs]))
    return obs, infos


class SingleEnvBatch(Wrapper, EnvBatch):
  def __init__(self, env):
    super(SingleEnvBatch, self).__init__(env)
    self.observation_space = SpaceBatch([self.env.observation_space])
    self.action_space = SpaceBatch([self.env.action_space])

  @property
  def nenvs(self):
    return 1

  @property
  def envs(self):
    return [self.env]

  def step(self, actions):
    self._check_actions(actions)
    ob, rew, done, info = self.env.step(actions[0])
    if done:
      ob = self.env.reset()
    return ob[None], np.expand_dims(rew, 0), np.expand_dims(done, 0), [info]

  def reset(self):
    obs, info = self.env.reset()
    return obs[None], [info]


def worker(connection, make_env_function,
           send_spaces=True, **make_env_kwargs):
  # Adapted from SubprocVecEnv github.com/openai/baselines
  env = make_env_function(**make_env_kwargs)
  if send_spaces:
    connection.send((env.observation_space, env.action_space))
  while True:
    try:
      cmd, args = connection.recv()
    except EOFError:
      return
    if cmd == "step":
      action = args
      ob, rew, terminated, truncated, info = env.step(action)
      if terminated or truncated:
        ob, _ = env.reset()
      connection.send((ob, rew, terminated, truncated, info))
    elif cmd == "reset":
      kwargs = args
      ob, info = env.reset()
      connection.send((ob, info))
    elif cmd == "render":
      return env.render()
    elif cmd == "close":
      env.close()
      connection.close()
      break
    else:
      raise NotImplementedError("Unknown command %s" % cmd)


class ParallelEnvBatch(EnvBatch):
  """
  An abstract batch of environments.
  """
  def __init__(self, make_env, make_env_kwargs, nenvs=None):
    if not isinstance(make_env_kwargs, (tuple, list, np.ndarray)):
      if nenvs is not None:
        raise ValueError("when make_env_kwargs is not an instance of "
                         "(tuple, list, np.ndarray) nenvs must not be None")
      make_env_kwargs = [make_env_kwargs for _ in range(nenvs)]
    self._nenvs = nenvs if nenvs is not None else len(make_env_kwargs)
    self._parent_connections, self._worker_connections = zip(*[
      Pipe() for _ in range(self._nenvs)
    ])
    self._processes = [
        Process(
          target=worker,
          args=(connection, make_env),
          kwargs=kwargs,
          daemon=True
        )
        for connection, kwargs
        in zip(self._worker_connections, make_env_kwargs)
    ]
    for p in self._processes:
      p.start()
    self._closed = False

    observation_spaces, action_spaces = [], []
    for conn in self._parent_connections:
      ob_space, ac_space = conn.recv()
      observation_spaces.append(ob_space)
      action_spaces.append(ac_space)
    self.observation_space = SpaceBatch(observation_spaces)
    self.action_space = SpaceBatch(action_spaces)

  @property
  def nenvs(self):
    return self._nenvs

  def step(self, actions):
    self._check_actions(actions)
    for conn, a in zip(self._parent_connections, actions):
      conn.send(("step", a))
    results = [conn.recv() for conn in self._parent_connections]
    obs, rews, terminations, truncations, infos = zip(*results)
    return (np.stack(obs), np.stack(rews),
            np.stack(terminations), np.stack(truncations),
            infos)

  def reset(self, **kwargs):
    for conn in self._parent_connections:
      conn.send(("reset", kwargs))
    results = [conn.recv() for conn in self._parent_connections]
    obs, infos = zip(*results)
    return np.stack(obs), infos

  def close(self):
    if self._closed:
      return
    for conn in self._parent_connections:
      conn.send(("close", None))
    for p in self._processes:
      p.join()
    self._closed = True

  def render(self):
    for conn in self._parent_connections:
      conn.send(("render", {}))
    frames = [conn.recv() for conn in self._parent_connections]
    return np.stack(frames)
