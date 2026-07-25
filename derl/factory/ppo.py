""" Implements PPO factory. """
from torch.optim import Adam
from derl.alg.common import Trainer
from derl.anneal import LinearAnneal
from derl.factory.factory import Factory
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.alg.ppo import PPO
from derl.runners.onpolicy import make_ppo_runner


class PPOFactory(Factory):
  """ Proximal Policy Optimization factory. """
  @staticmethod
  def get_argdict(args_type="atari"):
    defaults = {
        "atari": {
            "num-train-steps": 10e6,
            "num-recordings": 10,
            "nenvs": 8,
            "num-runner-steps": 128,
            "gamma": 0.99,
            "lambda_": 0.95,
            "num-epochs": 3,
            "num-minibatches": 4,
            "cliprange": 0.1,
            "value-loss-coef": 0.25,
            "entropy-coef": 0.01,
            "max-grad-norm": 0.5,
            "lr": 2.5e-4,
            "optimizer-epsilon": 1e-5,
        },
        "mujoco": {
            "num-train-steps": 1e6,
            "num-recordings": 10,
            "nenvs": dict(type=int, default=None),
            "num-runner-steps": 2048,
            "gamma": 0.99,
            "lambda_": 0.95,
            "num-epochs": 10,
            "num-minibatches": 32,
            "cliprange": 0.2,
            "value-loss-coef": 0.25,
            "entropy-coef": 0.,
            "max-grad-norm": 0.5,
            "lr": 3e-4,
            "optimizer-epsilon": 1e-5,
        }
    }
    return defaults.get(args_type)

  def make_runner(self, env, model=None, nlogs=1e5, **kwargs):
    self.config |= kwargs
    if model is None:
      model = make_model(env.observation_space, env.action_space, 1)
    policy = ActorCriticPolicy(model)
    runner = make_ppo_runner(env, policy, self.num_runner_steps,
                             self.num_train_steps, nlogs=nlogs,
                             gamma=self.gamma,
                             lambda_=self.lambda_,
                             num_epochs=self.num_epochs,
                             num_minibatches=self.num_minibatches)
    return runner

  def make_trainer(self, runner, **kwargs):
    self.config |= kwargs
    lr = LinearAnneal(self.lr, self.num_train_steps, name="lr")
    params = runner.policy.model.parameters()
    optimizer_kwargs = {"params": params, "lr": lr.get_tensor(),
                        "eps": self.optimizer_epsilon}
    optimizer = Adam(**optimizer_kwargs)
    return Trainer(optimizer, anneals=[lr],
                   max_grad_norm=self.max_grad_norm)

  def make_alg(self, runner, trainer, **kwargs):
    self.config |= kwargs
    ppo = PPO(runner, trainer,
              value_loss_coef=self.value_loss_coef,
              entropy_coef=self.entropy_coef,
              cliprange=self.cliprange)
    return ppo
