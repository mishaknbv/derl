""" Wrapper for writing summaries. """
from collections import deque
import sys
from gymnasium import Wrapper
import numpy as np
import torch
from derl import summary


class VideoRecording(Wrapper):
  """ Records the interactions and saves them as a video. """
  def __init__(self, env, recording_period, prefix=None, fps=25):
    super().__init__(env)
    self.recording_period = recording_period
    self.prefix = prefix or self.env.spec.id
    self.fps = fps
    self.step_count = 0
    self.last_recording = -sys.maxsize
    self.frames = []
    self.had_ended_episodes = np.zeros(
        getattr(self.env.unwrapped, "nenvs", 1), bool)

  @classmethod
  def make(cls, env, nlogs, nsteps):
    """ Creates an instance that will write nlogs over nsteps. """
    return cls(env, nsteps // nlogs + 1)

  def save_video(self):
    """ Saves the video of the last frames. """
    frames = torch.tensor(np.asarray(self.frames))
    if frames.ndim == 4:
      frames = frames.permute(0, 3, 1, 2).unsqueeze(0)
    elif frames.ndim == 5:
      frames = frames.permute(1, 0, 4, 2, 3)
    summary.add_video(self.prefix, frames,
                      fps=self.fps, global_step=self.step_count)

  def step(self, action):
    obs, rew, terminated, truncated, info = self.env.step(action)
    self.frames.append(self.env.render())
    if hasattr(self.unwrapped, "nenvs"):
      resets = np.asarray([
          info[i].get("real_done", terminated[i] or truncated[i])
          for i in range(self.unwrapped.nenvs)
      ])
    else:
      resets = np.asarray([info.get("real_done", terminated or truncated)])

    self.had_ended_episodes |= resets
    if (np.all(self.had_ended_episodes)
        and self.step_count - self.last_recording >= self.recording_period):
      self.save_video()
      self.last_recording = self.step_count
      self.had_ended_episodes.fill(False)
    if np.all(self.had_ended_episodes):
      self.frames.clear()
      self.had_ended_episodes.fill(False)
    self.step_count += self.had_ended_episodes.shape[0]
    return obs, rew, terminated, truncated, info



class RewardSummarizer:
  """ Summarizes rewards received from environment. """
  def __init__(self, nenvs, prefix, running_mean_size=100):
    self.prefix = prefix
    self.step_count = 0
    self.had_ended_episodes = np.zeros(nenvs, dtype=bool)
    self.rewards = np.zeros(nenvs)
    self.episode_lengths = np.zeros(nenvs)
    self.reward_queues = [deque([], maxlen=running_mean_size)
                          for _ in range(nenvs)]

  def should_add_summaries(self):
    """ Returns `True` if it is time to write summaries. """
    return summary.should_record() and np.all(self.had_ended_episodes)

  def add_summaries(self):
    """ Writes summaries. """
    summaries = dict(
        total_reward=np.mean([q[-1] for q in self.reward_queues]),
        episode_length=np.mean(self.episode_lengths),
        min_reward=min(q[-1] for q in self.reward_queues),
        max_reward=max(q[-1] for q in self.reward_queues),
    )
    summaries[f"reward_mean_{self.reward_queues[0].maxlen}"] = \
        np.mean([np.mean(q) for q in self.reward_queues])

    for key, val in summaries.items():
      summary.add_scalar(f"{self.prefix}/{key}", val,
                         global_step=self.step_count)

  def step(self, rewards, resets):
    """ Takes statistics from last env step and tries to add summaries.  """
    self.rewards += rewards
    self.episode_lengths[~self.had_ended_episodes] += 1
    for i, in zip(*resets.nonzero()):
      self.reward_queues[i].append(self.rewards[i])
      self.rewards[i] = 0
      self.had_ended_episodes[i] = True

    self.step_count += self.rewards.shape[0]
    if self.should_add_summaries():
      self.add_summaries()
      self.episode_lengths.fill(0)
      self.had_ended_episodes.fill(False)

  def reset(self):
    """ Resets the reward summarizer. """
    for i, queue in enumerate(self.reward_queues):
      if self.episode_lengths[i] and not self.had_ended_episodes[i]:
        queue.append(self.rewards[i])
        self.rewards[i] = 0
        self.had_ended_episodes[i] = True
    if self.should_add_summaries():
      self.add_summaries()
      self.episode_lengths.fill(0)
      self.had_ended_episodes.fill(False)


class Summarize(Wrapper):
  """ Writes env summaries."""
  def __init__(self, env, summarizer):
    super().__init__(env)
    self.summarizer = summarizer

  @classmethod
  def reward_summarizer(cls, env, prefix=None, running_mean_size=100):
    """ Creates an instance with reward summarizer. """
    nenvs = getattr(env.unwrapped, "nenvs", 1)
    prefix = prefix if prefix is not None else env.spec.id
    summarizer = RewardSummarizer(nenvs, prefix,
                                  running_mean_size=running_mean_size)
    return cls(env, summarizer)

  def step(self, action):
    obs, rew, terminated, truncated, info = self.env.step(action)

    info_collection = [info] if isinstance(info, dict) else info
    done_collection = ([terminated | truncated] if isinstance(terminated, bool)
                       else terminated | truncated)
    resets = np.asarray([info.get("real_done", done_collection[i])
                         for i, info in enumerate(info_collection)])
    self.summarizer.step(rew, resets)

    return obs, rew, terminated, truncated, info

  def reset(self, **kwargs):
    self.summarizer.reset()
    return self.env.reset(**kwargs)
