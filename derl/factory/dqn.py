"""Implements Deep Q-Learning Learner."""

from torch.optim import RMSprop

from derl.alg.common import Trainer
from derl.alg.dqn import DQN
from derl.anneal import LinearAnneal
from derl.factory.factory import Factory
from derl.models import NatureCNNModel
from derl.policies import EpsilonGreedyPolicy
from derl.runners.experience_replay import make_dqn_runner


class DQNFactory(Factory):
    """Deep Q-Learning Learner."""

    @staticmethod
    def get_argdict(args_type="atari"):
        return {
            "atari": {
                "num-train-steps": 200e6,
                "num-recordings": 200,
                "no-distributional": dict(action="store_false", dest="distributional"),
                "num-quantiles": 200,
                "no-dueling": dict(action="store_false", dest="dueling"),
                "no-noisy": dict(action="store_false", dest="noisy"),
                "exploration-epsilon-start": 1.0,
                "exploration-epsilon-end": 0.01,
                "exploration-end-step": int(1e6),
                "storage-size": int(1e6),
                "storage-init-size": int(50e3),
                "no-prioritized": dict(action="store_false", dest="prioritized"),
                "per-alpha": 0.6,
                "per-beta": dict(type=float, default=(0.4, 1.0), nargs=2),
                "steps-per-sample": 4,
                "batch-size": 32,
                "nstep": 3,
                "lr": 2.5e-4,
                "optimizer-decay": 0.95,
                "optimizer-momentum": 0.0,
                "optimizer-epsilon": 0.01,
                "gamma": 0.99,
                "target-update-period": int(10e3),
                "no-double": dict(action="store_false", dest="double"),
            },
        }.get(args_type)

    def make_model(self, env, init_fn=None, **kwargs):
        """Creates Nature-DQN model for a given env."""
        self.config |= kwargs
        model_kwargs = dict(
            noisy=self.noisy,
            dueling=self.dueling,
            num_quantiles=getattr(self.config, "num_quantiles", None),
        )
        if not self.distributional:
            model_kwargs.pop("num_quantiles")
        return NatureCNNModel(
            input_shape=env.observation_space.shape,
            output_units=env.action_space.n,
            **model_kwargs,
            init_fn=init_fn,
        )

    def make_runner(self, env, model=None, nlogs=1e5, **kwargs):
        self.config |= kwargs
        if model is None:
            model = self.make_model(env, noisy=self.noisy, dueling=self.dueling)
        epsilon = 0.0
        anneals = []
        start, nstep, end = (
            self.exploration_epsilon_start,
            self.exploration_end_step,
            self.exploration_epsilon_end,
        )
        if not self.noisy:
            epsilon_anneal = LinearAnneal(start, nstep, end, name="exploration_epsilon")
            epsilon = epsilon_anneal.get_tensor()
            anneals.append(epsilon_anneal)
        policy = (
            EpsilonGreedyPolicy.quantile(model, epsilon)
            if self.distributional
            else EpsilonGreedyPolicy(model, epsilon)
        )
        runner_kwargs = dict(
            storage_size=self.storage_size,
            storage_init_size=self.storage_init_size,
            batch_size=self.batch_size,
            steps_per_sample=self.steps_per_sample,
            nstep=self.nstep,
            prioritized=self.prioritized,
        )
        runner_kwargs["alpha"] = self.per_alpha
        runner_kwargs["beta"] = self.per_beta
        runner = make_dqn_runner(
            env,
            policy,
            self.num_train_steps,
            anneals=anneals,
            nlogs=nlogs,
            **runner_kwargs,
        )
        return runner

    def make_trainer(self, runner, **kwargs):
        self.config |= kwargs
        model = runner.policy.model
        optimizer_kwargs = {
            "alpha": self.optimizer_decay,
            "momentum": self.optimizer_momentum,
            "eps": self.optimizer_epsilon,
        }
        optimizer = RMSprop(model.parameters(), self.lr, **optimizer_kwargs)
        return Trainer(optimizer)

    def make_alg(self, runner, trainer, **kwargs):
        self.config |= kwargs
        alg = DQN(
            runner,
            trainer,
            gamma=self.gamma,
            target_update_period=self.target_update_period,
            double=self.double,
        )
        return alg
