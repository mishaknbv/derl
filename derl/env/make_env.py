"""Creates environments with standard wrappers."""

import ale_py
import gymnasium as gym

from .atari_wrappers import (
    ClipReward,
    EpisodicLife,
    FireReset,
    ImagePreprocessing,
    MaxBetweenFrames,
    QueueFrames,
    SkipFrames,
    StartWithRandomActions,
)
from .env_batch import ParallelEnvBatch
from .mujoco_wrappers import Normalize, TanhRangeActions
from .summarize import Summarize, VideoRecording

gym.register_envs(ale_py)


def list_envs(env_type=None):
    """Returns list of envs ids of given type."""
    impossible_roms = {"maze_craze", "joust", "warlords", "combat"}
    all_atari_games = {
        env_spec.kwargs["game"]
        for env_spec in gym.registry.values()
        if isinstance(env_spec.entry_point, str)
        and "ale_py" in env_spec.entry_point
        and env_spec.kwargs["game"] not in impossible_roms
    }
    ids = {
        "atari": [
            "".join(c.capitalize() for c in g.split("_")) for g in all_atari_games
        ],
        "mujoco": [
            "Reacher",
            "Pusher",
            "Thrower",
            "Striker",
            "InvertedPendulum",
            "InvertedDoublePendulum",
            "HalfCheetah",
            "Hopper",
            "Swimmer",
            "Walker2d",
            "Ant",
            "Humanoid",
            "HumanoidStandup",
        ],
        "classic-control": [
            "Acrobot",
            "CartPole",
            "MountainCarContinuous",
            "MountainCar",
            "Pendulum",
        ],
    }
    return ids[env_type] if env_type is not None else ids


def is_atari_id(env_id):
    """Returns True if env_id corresponds to an Atari env."""
    env_id = env_id[: env_id.rfind("-")]
    for postfix in ("Deterministic", "NoFrameskip"):
        env_id = env_id.removesuffix(postfix)

    atari_envs = set(list_envs("atari"))
    return env_id in atari_envs


def is_mujoco_id(env_id):
    """Returns True if env_id corresponds to MuJoCo env."""
    env_id = "".join(env_id.split("-")[:-1])
    mujoco_ids = set(list_envs("mujoco"))
    return env_id in mujoco_ids


def is_classic_control_id(env_id):
    """Returns True if env_id corresponds to classic control env."""
    env_id = "".join(env_id.split("-")[:-1])
    return env_id in set(list_envs("classic-control"))


def get_env_type(env_id):
    """Returns the type of environment."""
    env_id = "".join(env_id.split("-")[:-1])
    env_id = env_id.removesuffix("NoFrameskip")
    for key, envs in list_envs().items():
        if env_id in envs:
            return key
    raise ValueError(f"unknown env_type for {env_id=}")


def get_seed(nenvs=None, seed=None):
    """Returns seed(s) for specified number of envs."""
    if nenvs is None and seed is not None and not isinstance(seed, int):
        raise ValueError(
            f"when nenvs is None seed must be None or an int, got type {type(seed)}"
        )
    if nenvs is None:
        return seed or 0
    if isinstance(seed, (list, tuple)):
        if len(seed) != nenvs:
            raise ValueError(f"seed must have length {nenvs} but has {len(seed)}")
        return seed
    if seed is None:
        seed = list(range(nenvs))
    elif isinstance(seed, int):
        seed = [seed] * nenvs
    else:
        raise ValueError(f"invalid seed: {seed}")
    return seed


def nature_dqn_env(
    env_id,
    nenvs=None,
    seed=None,
    render_mode="rgb_array",
    max_num_frames_per_episode=108_000,
    repeat_action_prob=0.0,
    **kwargs,
):
    """Wraps env as in Nature DQN paper."""
    assert is_atari_id(env_id)
    if "NoFrameskip" not in env_id:
        raise ValueError(f"env_id must have 'NoFrameskip' but is {env_id}")
    seed = get_seed(nenvs)
    if nenvs is not None:
        env = ParallelEnvBatch(
            nature_dqn_env,
            make_env_kwargs=[
                dict(
                    env_id=env_id,
                    seed=s,
                    max_num_frames_per_episode=max_num_frames_per_episode,
                    repeat_action_prob=repeat_action_prob,
                )
                | kwargs
                | {"summarize": False, "recording_period": None}
                for s in seed
            ],
        )
        return nature_dqn_wrap(env, prefix=env_id, **kwargs)

    ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    env = gym.make(
        env_id,
        max_num_frames_per_episode=max_num_frames_per_episode,
        repeat_action_probability=repeat_action_prob,
        render_mode=render_mode,
    )
    env.action_space.seed(seed)
    return nature_dqn_wrap(env, **kwargs)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def nature_dqn_wrap(
    env,
    prefix=None,
    recording_period=None,
    recording_filepath=None,
    summarize=True,
    episodic_life=True,
    max_random_actions=30,
    num_queue_frames=4,
):
    """Wraps given env as in nature DQN paper."""
    if recording_period is not None:
        env = VideoRecording(
            env,
            recording_period,
            prefix=prefix or env.spec.id,
            output_filepath=recording_filepath,
        )
    if summarize:
        env = Summarize.reward_summarizer(env, prefix=prefix or env.spec.id)
    if hasattr(env.unwrapped, "nenvs"):
        return env
    if episodic_life:
        env = EpisodicLife(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = StartWithRandomActions(env, max_random_actions=max_random_actions)
    env = MaxBetweenFrames(env)
    env = SkipFrames(env, 4)
    env = ImagePreprocessing(env, width=84, height=84, grayscale=True)
    env = QueueFrames(env, num_queue_frames)
    env = ClipReward(env)
    return env


# pylint: enable=too-many-arguments


def mujoco_env(env_id, nenvs=None, seed=None, render_mode="rgb_array", **kwargs):
    """Creates and wraps MuJoCo env."""
    assert is_mujoco_id(env_id)
    seed = get_seed(nenvs, seed)
    if nenvs is not None:
        env = ParallelEnvBatch(
            mujoco_env,
            [
                dict(
                    seed=s,
                    recording_period=None,
                    render_mode=render_mode,
                    summarize=False,
                    normalize_obs=False,
                    normalize_ret=False,
                    tanh_range_actions=False,
                )
                for s in seed
            ],
        )
        return mujoco_wrap(env, **kwargs)

    env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(seed)
    return mujoco_wrap(env, **kwargs)


def mujoco_wrap(
    env,
    recording_period=None,
    summarize=True,
    normalize_obs=True,
    normalize_ret=True,
    tanh_range_actions=False,
):
    """Wraps given env as a mujoco env."""
    if recording_period is not None:
        env = VideoRecording(env, recording_period)
    if summarize:
        env = Summarize.reward_summarizer(env)
    if normalize_obs or normalize_ret:
        env = Normalize(env, obs=normalize_obs, ret=normalize_ret)
    if tanh_range_actions:
        env = TanhRangeActions(env)
    return env


def make(
    env_id, nenvs=None, seed=None, recording_period=None, summarize=True, **kwargs
):
    """Creates env with standard wrappers."""
    if is_atari_id(env_id):
        return nature_dqn_env(
            env_id,
            nenvs,
            seed=seed,
            recording_period=recording_period,
            summarize=summarize,
            **kwargs,
        )
    if is_mujoco_id(env_id):
        return mujoco_env(
            env_id,
            nenvs,
            seed=seed,
            recording_period=recording_period,
            summarize=summarize,
            **kwargs,
        )

    seed = get_seed(nenvs, seed)
    if nenvs is not None:
        env = ParallelEnvBatch(
            make,
            [
                dict(env_id=env_id, seed=s)
                | kwargs
                | {"summarize": False, "recording_period": None}
                for s in seed
            ],
        )
        if recording_period is not None:
            env = VideoRecording(env, prefix=env_id, recording_period=recording_period)
        if summarize:
            env = Summarize.reward_summarizer(env, prefix=env_id)
        return env

    render_mode = kwargs.pop("render_mode", "rgb_array")
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    env.action_space.seed(seed)
    return env
