# pylint: skip-file
""" RND factory. """
from derl.factory.ppo import PPOFactory
from derl.alg.rnd import RNDModel, make_rnd_runner, RND
from derl.env.summarize import VideoRecording
from derl.policies import ActorCriticPolicy


class RNDFactory(PPOFactory):
  """ RND factory. """
  @staticmethod
  def get_argdict(args_type="atari"):
    argdict = {
        "atari": {
            "num-train-steps": 1.97e9,
            "nenvs": 128,
            "num-recordings": 100,
            "num-runner-steps": 128,
            "extrinsic-gamma": 0.999,
            "intrinsic-gamma": 0.99,
            "extrinsic-reward-coef": 2.,
            "intrinsic-reward-coef": 1.,
            "lambda_": 0.95,
            "num-epochs": 4,
            "num-minibatches": 4,
            "cliprange": 0.1,
            "lr": 1e-4,
            "optimizer-epsilon": 1e-5,
            "entropy-coef": 0.001,
            "prob-distill": 0.25,
            "max-grad-norm": dict(type=float, default=None),
        }
    }
    return argdict[args_type]

  def make_env(self, env_id, **kwargs):
    self.config |= kwargs
    env = super().make_env(
        env_id,
        nenvs=self.nenvs,
        max_num_frames_per_episode=18_000,
        episodic_life=False,
        max_random_actions=0,
        repeat_action_prob=0.25,
    )
    return env

  def make_runner(self, env, model=None, nlogs=1e5, **kwargs):
    self.config |= kwargs
    if model is None:
      model = RNDModel(env.action_space.n, env.observation_space.shape)
    policy = ActorCriticPolicy(model)
    runner = make_rnd_runner(env, policy, self.num_runner_steps,
                             self.num_train_steps, nlogs=nlogs,
                             extrinsic_gamma=self.extrinsic_gamma,
                             intrinsic_gamma=self.intrinsic_gamma,
                             extrinsic_coef=self.extrinsic_reward_coef,
                             intrinsic_coef=self.intrinsic_reward_coef,
                             lambda_=self.lambda_,
                             num_epochs=self.num_epochs,
                             num_minibatches=self.num_minibatches)
    return runner

  def make_alg(self, runner, trainer, **kwargs):
    self.config |= kwargs | {"rnd": False}
    rnd = RND(runner, trainer,
              intrinsic_gamma=self.intrinsic_gamma,
              cliprange=self.cliprange,
              entropy_coef=self.entropy_coef,
              prob_distill=self.prob_distill)
    return rnd
