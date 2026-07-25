""" Advantage Actor-Critic Learner. """
from torch.optim import RMSprop
from derl.alg.a2c import A2C
from derl.alg.common import Trainer
from derl.anneal import LinearAnneal
from derl.factory.factory import Factory
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.runners.env_runner import EnvRunner
from derl.runners.onpolicy import TransformInteractions
from derl.runners.summary import PeriodicSummaries
from derl.runners.trajectory_transforms import GAE, MergeTimeBatch


class A2CFactory(Factory):
  """ Advantage Actor-Critic Learner. """
  @staticmethod
  def get_argdict(args_type="atari"):
    return {
        "atari": {
            "nenvs": 8,
            "num-train-steps": 10e6,
            "num-recordings": 10,
            "num-runner-steps": 5,
            "gamma": 0.99,
            "lambda_": 1.,
            "normalize-gae": dict(action="store_true"),
            "lr": 7e-4,
            "optimizer-alpha": 0.99,
            "optimizer-epsilon": 1e-5,
            "value-loss-coef": 0.5,
            "entropy-coef": 0.01,
            "max-grad-norm": 1.5,
        }
    }.get(args_type)

  def make_runner(self, env, model=None, nlogs=1e5, **kwargs):
    self.config |= kwargs
    if model is None:
      model = make_model(env.observation_space, env.action_space, 1)
    policy = ActorCriticPolicy(model)
    runner = EnvRunner(env, policy, self.num_runner_steps,
                       nsteps=self.num_train_steps)
    runner = PeriodicSummaries.make_with_nlogs(runner, nlogs)
    transforms = [GAE(policy, gamma=self.gamma, lambda_=self.lambda_,
                      normalize=self.normalize_gae)]
    if hasattr(env.unwrapped, "nenvs"):
      transforms.append(MergeTimeBatch())
    runner = TransformInteractions(runner, transforms)
    return runner

  def make_trainer(self, runner, **kwargs):
    self.config |= kwargs
    lr = LinearAnneal(self.lr, self.num_train_steps, 0., name="lr")
    optimizer_kwargs = {
        "alpha": self.optimizer_alpha,
        "eps": self.optimizer_epsilon
    }
    optimizer = RMSprop(runner.policy.model.parameters(),
                        lr.get_tensor(), **optimizer_kwargs)
    trainer =  Trainer(optimizer, anneals=[lr],
                       max_grad_norm=self.max_grad_norm)
    return trainer

  def make_alg(self, runner, trainer, **kwargs):
    self.config |= kwargs
    return A2C(runner, trainer, value_loss_coef=self.value_loss_coef,
               entropy_coef=self.entropy_coef)
