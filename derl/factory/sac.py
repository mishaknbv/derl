""" Implements SAC factory. """
import torch
from torch.optim import Adam
from derl.factory.factory import Factory
from derl.models import  SACModel
from derl.policies import SACPolicy
from derl.alg.sac import SAC, SACTrainer
from derl.runners.experience_replay import make_mujoco_sac_runner


class SACFactory(Factory):
  """ Soft Actor-Critic factory. """

  @staticmethod
  def get_argdict(args_type="atari"):
    return {
        "mujoco": {
            "num-train-steps": 1e6,
            "num-recordings": 10,
            "storage-size": int(1e6),
            "storage-init-size": 1000,
            "batch-size": 256,
            "steps-per-sample": 1000,
            "num-storage-samples": 1000,
            "reward-scale": 1.,
            "gamma": 0.99,
            "target-update-period": 1,
            "target-update-coef": 0.005,
            "policy-lr": 3e-4,
            "qvalue-lr": 3e-4,
            "entropy-scale-lr": 3e-4,
        }
    }.get(args_type)

  def make_env(self, env_id, **kwargs):
    self.config |= kwargs
    return super().make_env(env_id,
                            normalize_obs=False,
                            normalize_ret=False,
                            tanh_range_actions=True)

  def make_runner(self, env, model=None, nlogs=1e5, **kwargs):
    self.config |= kwargs
    if model is None:
      model = SACModel.make(env.observation_space, env.action_space)
    policy = SACPolicy(model)
    runner = make_mujoco_sac_runner(
      env, policy,
      num_train_steps=self.num_train_steps,
      storage_size=self.storage_size,
      storage_init_size=self.storage_init_size,
      batch_size=self.batch_size,
      steps_per_sample=self.steps_per_sample,
      num_storage_samples=self.num_storage_samples
    )
    return runner

  def make_trainer(self, runner, **kwargs):
    self.config |= kwargs
    model = runner.policy.model
    policy_opt = Adam(model.policy.parameters(), self.policy_lr)
    entropy_scale_opt = Adam((torch.zeros((), requires_grad=True),),
                             self.entropy_scale_lr)
    qvalue_opts = [Adam(qv.parameters(), self.qvalue_lr)
                   for qv in model.qvalues]
    trainer = SACTrainer(policy_opt, entropy_scale_opt, qvalue_opts)
    return trainer

  def make_alg(self, runner, trainer, **kwargs):
    self.config |= kwargs
    log_entropy_scale = \
        trainer.entropy_scale_opt.param_groups[0]["params"][0]
    sac = SAC.make(runner, trainer, log_entropy_scale=log_entropy_scale,
                   target_update_coef=self.target_update_coef,
                   target_update_period=self.target_update_period,
                   gamma=self.gamma,
                   reward_scale=self.reward_scale)
    return sac
