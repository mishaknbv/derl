"""Random network distillation.

https://arxiv.org/abs/1810.12894
"""

from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from torch import nn
from tqdm import trange

from derl import summary
from derl.alg.ppo import PPO
from derl.env.mujoco_wrappers import Normalize, RunningMeanVar
from derl.models import NatureCNNBase, broadcast_inputs, get_device
from derl.runners.env_runner import EnvRunner
from derl.runners.onpolicy import IterateWithMinibatches, TransformInteractions
from derl.runners.summary import PeriodicSummaries
from derl.runners.trajectory_transforms import (
    GAE,
    MergeTimeBatch,
)


class IntrinsicReward(Normalize):
    """Intrinsic reward based on predictor error."""

    def __init__(
        self,
        env,
        predictor,
        target,
        normalize=True,
        clipobs=5.0,
        intrinsic_gamma=0.99,
        eps=1e-8,
    ):
        super().__init__(
            env,
            obs=normalize,
            ret=normalize,
            clipobs=clipobs,
            cliprew=np.inf,
            gamma=intrinsic_gamma,
        )
        height, width, _ = self.observation_space.shape
        self.obs_rmv = RunningMeanVar(shape=(height, width, 1)) if normalize else None
        self.predictor = predictor
        self.target = target

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        with torch.no_grad():
            nobs = self.observation(obs.astype(np.float32)[..., -1:])
            intrinsic_rew = (
                torch.mean(torch.square(self.predictor(nobs) - self.target(nobs)), -1)
                .cpu()
                .numpy()
            )
        self.ret = self.ret * self.gamma + intrinsic_rew
        if self.ret_rmv:
            self.ret_rmv.update(self.ret)
            intrinsic_rew = intrinsic_rew / np.sqrt(self.ret_rmv.var + self.eps)
        return obs, (rew, intrinsic_rew), terminated, truncated, info

    def reset(self, **kwargs):
        self.ret.fill(0.0)
        obs, info = self.env.reset(**kwargs)
        if self.obs_rmv:
            self.obs_rmv.update(obs.astype(np.float32)[..., -1:])
        return obs, info


def reinitialize(model, init_fn=None) -> int:
    """Reinitializes the model."""
    if init_fn is not None:
        model.apply(init_fn)
        return -1
    nresets = 0
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
            nresets += 1
    return nresets


def replace_module(model, old_type, new_type):
    """Replaces module in a model."""
    for name, child in model.named_children():
        if isinstance(child, old_type):
            setattr(model, name, new_type())
        else:
            replace_module(child, old_type, new_type)


class PredictorModel(nn.Module):
    """Predictor model trained to distill random network."""

    def __init__(
        self, base, observation_sample, num_extra_layers=2, activation=nn.ReLU
    ):
        super().__init__()
        self.base = base
        with torch.no_grad():
            output_shape = self.base(observation_sample).shape
        extra_layers = list(
            chain.from_iterable(
                (activation(), nn.Linear(output_shape[-1], output_shape[-1]))
                for _ in range(num_extra_layers)
            )
        )
        self.extra_layers = nn.Sequential(*extra_layers)
        self.to(next(base.parameters()).device)

    def forward(self, *inputs):
        """Forward pass of the network."""
        return self.extra_layers(self.base(*inputs))


class RNDModel(nn.Module):
    """Random network distillation module."""

    def __init__(
        self,
        num_actions,
        input_shape=(84, 84, 4),
        base_outputs=512,
        hidden_units=512,
        init_fn=None,
    ):
        super().__init__()
        self.base = NatureCNNBase(input_shape, output_features=base_outputs)
        self.base_addon = nn.Sequential(
            nn.ReLU(),
            nn.Linear(base_outputs, hidden_units),
            nn.ReLU(),
        )
        self.residuals = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
                for _ in range(2)
            ]
        )
        self.policy = nn.Linear(hidden_units, num_actions)
        self.value = nn.Linear(hidden_units, 2)
        self.init_fn = init_fn
        if self.init_fn:
            self.apply(self.init_fn)
        self.to(get_device())

    @broadcast_inputs(ndims=4)
    def forward(self, inputs):
        """Forward pass of the module."""
        x = self.base_addon(self.base(inputs))
        policy, values = x + self.residuals[0](x), x + self.residuals[1](x)
        policy = self.policy(policy)
        values = self.value(values)
        return policy, values


class RNDGAE(GAE):
    """GAE for RND."""

    def __init__(
        self,
        policy,
        extrinsic_gamma=0.999,
        intrinsic_gamma=0.99,
        extrinsic_coef=2.0,
        intrinsic_coef=1.0,
        **kwargs,
    ):
        gamma = np.asarray([extrinsic_gamma, intrinsic_gamma])
        super().__init__(policy, gamma=gamma, **kwargs)
        self.extrinsic_gamma = extrinsic_gamma
        self.intrinsic_gamma = intrinsic_gamma
        self.extrinsic_coef = extrinsic_coef
        self.intrinsic_coef = intrinsic_coef

    def __call__(self, trajectory):
        trajectory["rewards"] = np.moveaxis(trajectory["rewards"], 1, -1)
        trajectory["terminations"] = np.stack(
            [trajectory["terminations"], np.zeros_like(trajectory["terminations"])], -1
        )
        trajectory["truncations"] = np.stack(
            [trajectory["truncations"], trajectory["truncations"]], -1
        )
        advantages, value_targets = super().__call__(trajectory)
        trajectory["advantages"] = (
            self.extrinsic_coef * advantages[..., 0]
            + self.intrinsic_coef * advantages[..., 1]
        )
        return advantages, value_targets


def rnd_runner_wrap(
    runner,
    num_epochs=3,
    num_minibatches=4,
    **gae_kwargs,
):
    """Wrapps given runner for RND training."""
    env, policy = runner.env, runner.policy
    transforms = [RNDGAE(policy, **gae_kwargs, normalize=False)]
    if not policy.is_recurrent() and getattr(env.unwrapped, "nenvs", None):
        transforms.append(MergeTimeBatch(check_shape=False))
    runner = TransformInteractions(runner, transforms)
    runner = IterateWithMinibatches(
        runner, num_epochs, num_minibatches, shuffle_before_epoch=False
    )
    # There is no advantage normalization in the reference implementation.
    # runner = TransformInteractions(runner, [NormalizeAdvantages()])
    return runner


def make_rnd_runner(env, policy, horizon, nsteps, nlogs=1e5, **wrap_kwargs):
    """Creates and wraps env runner for RND training."""
    runner = EnvRunner(env, policy, horizon, nsteps)
    runner = PeriodicSummaries.make_with_nlogs(runner, nlogs)
    return rnd_runner_wrap(runner, **wrap_kwargs)


class RND(PPO):
    """Random Network Distillation alogrithm."""

    def __init__(
        self,
        runner,
        trainer,
        intrinsic_gamma=0.99,
        num_extra_layers=2,
        distill_lr=1e-3,
        distill_loss_coef=1.0,
        prob_distill=0.25,
        num_warmup_steps=100,
        **kwargs,
    ):
        super().__init__(runner, trainer, **kwargs)
        observation_shape = runner.env.observation_space.shape
        height, width, _ = observation_shape
        self.target = NatureCNNBase(
            input_shape=(height, width, 1), activation=nn.LeakyReLU
        ).to(get_device())
        self.predictor = PredictorModel(
            deepcopy(self.target),
            self.runner.env.observation_space.sample()[..., -1:],
            num_extra_layers=num_extra_layers,
        ).to(get_device())
        reinitialize(self.predictor, self.runner.policy.model.init_fn)
        reinitialize(self.target, self.runner.policy.model.init_fn)
        self.trainer.optimizer.add_param_group(
            {
                "params": self.predictor.parameters(),
                "lr": distill_lr,
            }
        )

        runner.unwrapped.env = IntrinsicReward(
            runner.unwrapped.env,
            self.predictor,
            self.target,
            intrinsic_gamma=intrinsic_gamma,
        )
        self.distill_loss_coef = distill_loss_coef
        self.prob_distill = prob_distill
        self.num_warmup_steps = num_warmup_steps

    def distill_loss(self, data):
        """Distillation loss between the predictor and the target."""
        nobs = self.runner.env.normalize_observation(
            data["next_observations"].astype(np.float32)[..., -1:]
        )
        model_preds = self.predictor(nobs)
        with torch.no_grad():
            target_preds = self.target(nobs)
        batch_size = nobs.shape[0]
        intrinsic_reward = torch.mean(
            torch.square(target_preds - model_preds).reshape(batch_size, -1), -1
        )

        sample = torch.rand(batch_size).to(intrinsic_reward.device)
        mask = sample < self.prob_distill
        distill_loss = torch.sum(mask * intrinsic_reward) / torch.maximum(
            torch.tensor(1.0), torch.sum(mask)
        )
        if summary.should_record():
            summary.add_scalar(
                f"{self.name}/distill_loss",
                distill_loss,
                global_step=self.loss_fn.call_count,
            )
        return distill_loss

    def loss(self, data):
        loss = super().loss(data)
        return loss + self.distill_loss_coef * self.distill_loss(data)

    def learn(self, model_dump_period=1.97e9 // 100, model_filename="model-{step}.pt"):
        runner = self.runner.run()
        for _ in trange(self.num_warmup_steps, leave=False):
            if next(runner, None) is None:
                break
        super().learn(model_dump_period, model_filename)
