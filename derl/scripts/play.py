"""Script to record a video of a trained agent interacting with an env."""

import argparse
import os
import re
import time
from collections import UserList

import numpy as np
import torch

# pylint: disable=no-member
import derl


def get_last_model(logdir, regex=r"^\w+(?:-(\d+))?\.pt$"):
    """Returns the filepath and step of the last model in the logdir.

    Returns `(None, None)` if no files in `logdir` match the regex.
    """
    pattern = re.compile(regex)
    result = None, None
    for fname in os.listdir(logdir):
        match = pattern.match(fname)
        if not match:
            continue
        if not pattern.groups:  # no step, must have a single match
            result = os.path.join(logdir, fname), None
            continue
        step = match.group(1)
        step = int(step) if step is not None else None
        if result[1] is None or (step is not None and step >= result[1]):
            result = os.path.join(logdir, fname), step
    return result


def unwrap_recording(env):
    """Unwraps env until a VideoRecording wrapper."""
    while not isinstance(env, derl.env.VideoRecording):
        env = env.env
    return env


class FramesList(UserList):
    """Frames list is not cleared."""

    def clear(self):
        pass


def record(env, policy, output_filepath, seed=0, nepisodes=1, fps=30):
    """Records a video of a policy acting in an environment."""
    recording = unwrap_recording(env)
    recording.output_filepath = output_filepath
    recording.frames = FramesList()
    recording.recording_period = float("inf")

    state = policy.get_initial_state(1)
    resets = np.zeros(1, bool)
    obs, _ = env.reset(seed=seed)
    while nepisodes > 0:
        act = policy.act(obs, state, resets)
        obs, _, terminated, truncated, info = env.step(act["actions"])
        state = act.get("policy_state", None)
        resets[:] = terminated | truncated
        if info.get("real_done", terminated | truncated):
            nepisodes -= 1
            if nepisodes > 0:
                obs, _ = env.reset()

    recording.save_video(fps=fps)
    print(f"Wrote {len(recording.frames)} frames  to {output_filepath}")


def is_new_model(model_filepath, last):
    """Returns `True` if the model is newer than the last processed one."""
    if last is None:
        return True
    last_filepath, last_mtime = last
    if model_filepath != last_filepath:
        return True
    return os.path.getmtime(model_filepath) != last_mtime


def track(env_id, factory, logdir, args):
    """Keeps watching the logdir and records videos for new models."""
    env = factory.make_env(env_id, nenvs=None, num_recordings=args.nepisodes)
    policy = factory.make_runner(env).policy
    last = None
    print(f"Tracking {logdir} for new models (polling every {args.track_period}s) ...")
    try:
        while True:
            model_filepath, step = get_last_model(logdir)
            if model_filepath is None:
                time.sleep(args.track_period)
                continue
            if is_new_model(model_filepath, last):
                state_dict = torch.load(
                    model_filepath, map_location="cpu", weights_only=True
                )
                policy.model.load_state_dict(state_dict)
                print(f"Loading model {model_filepath}")
                filename = f"video-{step}.mp4" if step is not None else "video.mp4"
                output_filepath = os.path.join(logdir, filename)
                record(
                    env,
                    policy,
                    output_filepath,
                    seed=args.seed,
                    nepisodes=args.nepisodes,
                    fps=args.fps,
                )
                last = (model_filepath, os.path.getmtime(model_filepath))
            time.sleep(args.track_period)
    except KeyboardInterrupt:
        print("Stopped tracking.")


def get_parser():
    """Creates and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description="Records a video of an agent interacting with an environment."
    )
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nepisodes", type=int, default=1)
    parser.add_argument("--output-filepath")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--track-period", type=float, default=30.0)
    return parser


def play(env_id, factory, logdir, args=None):
    """Plays and records a video."""
    parser = get_parser()
    args = parser.parse_args(args)
    if args.track and args.model:
        raise ValueError("--model and --track cannot be used together")

    derl.summary.stop_recording()
    derl.summary.should_record = lambda *args, **kwargs: False
    if args.track:
        track(env_id, factory, logdir, args)
        return

    model_filepath = args.model
    step = None
    if model_filepath is None:
        model_filepath, step = get_last_model(logdir)
        if model_filepath is None:
            raise ValueError(f"no model files found in {logdir}")

    output_filepath = args.output_filepath
    if output_filepath is None:
        filename = f"video-{step}.mp4" if step is not None else "video.mp4"
        output_filepath = os.path.join(logdir, filename)

    env = factory.make_env(env_id, nenvs=None, num_recordings=args.nepisodes)
    runner = factory.make_runner(env, recurrent=args.recurrent)
    state_dict = torch.load(model_filepath, map_location="cpu", weights_only=True)
    print(f"Loading model {model_filepath}")
    runner.policy.model.load_state_dict(state_dict)

    record(
        env,
        runner.policy,
        output_filepath,
        seed=args.seed,
        nepisodes=args.nepisodes,
        fps=args.fps,
    )
