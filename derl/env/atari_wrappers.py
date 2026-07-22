""" Atari env wrappers. """
from collections import deque
import sys
import threading

import cv2
import gymnasium as gym
from gymnasium import spaces
import ale_py
import numpy as np
import torch
from derl import summary
cv2.ocl.setUseOpenCL(False)


class ObservationVideo(gym.Wrapper):
  """ Records the interactions and saves them as a video. """
  def __init__(self, env, recording_period, prefix=None,
               fps=25, multithread=True):
    super().__init__(env)
    self.recording_period = recording_period
    self.prefix = prefix or self.env.spec.id
    self.fps = fps
    self.multithread = multithread
    self.step_count = 0
    self.last_recording = -sys.maxsize
    self.obs_list = []
    self.had_ended_episodes = np.zeros(
        getattr(self.env.unwrapped, "nenvs", 1), bool)

  @classmethod
  def make(cls, env, nlogs, nsteps):
    """ Creates an instance that will write nlogs over nsteps. """
    return cls(env, nsteps // nlogs + 1)

  def save_video(self, frames=None):
    """ Saves the video of the last frames. """
    if frames is None:
      frames = self.obs_list
    frames = torch.tensor(np.asarray(frames))
    if frames.ndim == 4:
      frames = frames.permute(0, 3, 1, 2).unsqueeze(0)
    elif frames.ndim == 5:
      frames = frames.permute(1, 0, 4, 2, 3)
    summary.add_video(self.prefix, frames,
                      fps=self.fps, global_step=self.step_count)

  def step(self, action):
    obs, rew, terminated, truncated, info = self.env.step(action)
    if hasattr(self.unwrapped, "nenvs"):
      self.obs_list.append(np.asarray([
          info[i].get("raw_observation", obs[i])
          for i in range(self.unwrapped.nenvs)
      ]))
      resets = np.asarray([
          info[i].get("real_done", terminated[i] or truncated[i])
          for i in range(self.unwrapped.nenvs)
      ])
    else:
      self.obs_list.append(info.get("raw_observation", obs))
      resets = np.asarray([info.get("real_done", terminated or truncated)])

    self.had_ended_episodes |= resets
    if (np.all(self.had_ended_episodes)
        and self.step_count - self.last_recording >= self.recording_period):
      if self.multithread:
        threading.Thread(target=self.save_video,
                         args=(self.obs_list[:],), daemon=True).start()
      else:
        self.save_video()
      self.last_recording = self.step_count
      self.had_ended_episodes.fill(False)
    if np.all(self.had_ended_episodes):
      self.obs_list.clear()
      self.had_ended_episodes.fill(False)
    self.step_count += self.had_ended_episodes.shape[0]
    return obs, rew, terminated, truncated, info


class EpisodicLife(gym.Wrapper):
  """ Sets done flag to true when agent dies. """
  def __init__(self, env):
    super().__init__(env)
    self.lives = 0
    self.real_done = True

  def step(self, action):
    obs, rew, terminated, truncated, info = self.env.step(action)
    self.real_done = terminated
    info["real_done"] = terminated
    lives = self.env.unwrapped.ale.lives()
    if 0 < lives < self.lives:
      terminated = True
    self.lives = lives
    return obs, rew, terminated, truncated, info

  def reset(self, **kwargs):
    if self.real_done:
      obs, info = self.env.reset(**kwargs)
    else:
      obs, _, _, _, info = self.env.step(0)
    self.lives = self.env.unwrapped.ale.lives()
    return obs, info


class FireReset(gym.Wrapper):
  """ Makes fire action when reseting environment.

  Some environments are fixed until the agent makes the fire action,
  this wrapper makes this action so that the epsiode starts automatically.
  """
  def __init__(self, env):
    super().__init__(env)
    action_meanings = env.unwrapped.get_action_meanings()
    if len(action_meanings) < 3:
      raise ValueError(
          "env.unwrapped.get_action_meanings() must be of length >= 3"
          f"but is of length {len(action_meanings)}")
    if env.unwrapped.get_action_meanings()[1] != "FIRE":
      raise ValueError(
          "env.unwrapped.get_action_meanings() must have 'FIRE' "
          f"under index 1, but is {action_meanings}")

  def step(self, action):
    return self.env.step(action)

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    obs, _, terminated, truncated, _ = self.env.step(1)
    if terminated or truncated:
      self.env.reset(**kwargs)
    obs, _, terminated, truncated, info = self.env.step(2)
    if terminated or truncated:
      self.env.reset(**kwargs)
    return obs, info


class StartWithRandomActions(gym.Wrapper):
  """ Makes random number of random actions at the beginning of each
  episode. """
  def __init__(self, env, max_random_actions=30):
    super().__init__(env)
    self.max_random_actions = max_random_actions
    self.real_done = True

  def step(self, action):
    obs, rew, terminated, truncated, info = self.env.step(action)
    self.real_done = info.get("real_done", True)
    return obs, rew, terminated, truncated, info

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    if self.real_done:
      num_random_actions = self.env.action_space.np_random.integers(
          self.max_random_actions + 1)
      for _ in range(num_random_actions):
        action = self.env.action_space.sample()
        obs, _, _, _, info = self.env.step(action)
      self.real_done = False
    return obs, info


class ImagePreprocessing(gym.Wrapper):
  """ Preprocesses image-observations by possibly grayscaling and resizing. """
  def __init__(self, env, width=84, height=84, grayscale=True):
    super().__init__(env)
    self.width = width
    self.height = height
    self.grayscale = grayscale
    ospace = self.env.observation_space
    low, high, dtype = ospace.low.min(), ospace.high.max(), ospace.dtype
    if self.grayscale:
      self.observation_space = spaces.Box(low=low, high=high,
                                          shape=(width, height), dtype=dtype)
    else:
      obs_shape = (width, height) + self.observation_space.shape[2:]
      self.observation_space = spaces.Box(low=low, high=high,
                                          shape=obs_shape, dtype=dtype)

  def observation(self, observation):
    """ Performs image preprocessing. """
    if self.grayscale:
      observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, (self.width, self.height),
                             cv2.INTER_AREA)
    return observation

  def step(self, action):
    obs, rew, terminated, truncated, info = self.env.step(action)
    info["raw_observation"] = obs
    return self.observation(obs), rew, terminated, truncated, info

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    info["raw_observation"] = obs
    return self.observation(obs), info


class MaxBetweenFrames(gym.ObservationWrapper):
  """ Takes maximum between two subsequent frames. """
  def __init__(self, env):
    if (isinstance(env.unwrapped, ale_py.env.AtariEnv) and
        "NoFrameskip" not in env.spec.id):
      raise ValueError("MaxBetweenFrames requires NoFrameskip in atari env id")
    super().__init__(env)
    self.last_obs = None

  def observation(self, observation):
    obs = np.maximum(observation, self.last_obs)
    self.last_obs = observation
    return obs

  def reset(self, **kwargs):
    self.last_obs, info = self.env.reset(**kwargs)
    return self.last_obs, info


class QueueFrames(gym.ObservationWrapper):
  """ Queues specified number of frames together. """
  def __init__(self, env, nframes=4, concat=False):
    super().__init__(env)
    self.obs_queue = deque([], maxlen=nframes)
    self.concat = concat
    ospace = self.observation_space
    if self.concat:
      oshape = ospace.shape[:-1] + (ospace.shape[-1] * nframes,)
    else:
      oshape = ospace.shape + (nframes,)
    self.observation_space = spaces.Box(ospace.low.min(), ospace.high.max(),
                                        oshape, ospace.dtype)

  def observation(self, observation):
    self.obs_queue.append(observation)
    return (np.concatenate(self.obs_queue, -1) if self.concat
            else np.stack(self.obs_queue, -1))

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    for _ in range(self.obs_queue.maxlen - 1):
      self.obs_queue.append(obs)
    return self.observation(obs), info


class SkipFrames(gym.Wrapper):
  """ Performs the same action for several steps and returns the final result.
  """
  def __init__(self, env, nskip=4):
    super().__init__(env)
    if (isinstance(env.unwrapped, ale_py.env.AtariEnv) and
        "NoFrameskip" not in env.spec.id):
      raise ValueError("SkipFrames requires NoFrameskip in atari env id")
    self.nskip = nskip

  def step(self, action):
    total_reward = 0.0
    for _ in range(self.nskip):
      obs, rew, terminated, truncated, info = self.env.step(action)
      total_reward += rew
      if terminated or truncated:
        break
    return obs, total_reward, terminated, truncated, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class ClipReward(gym.RewardWrapper):
  """ Modifes reward to be in {-1, 0, 1} by taking sign of it. """
  def reward(self, reward):
    return np.sign(reward)
