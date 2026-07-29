""" Generic derl script to launch an algorithm. """
#!/usr/bin/env python3
from argparse import ArgumentParser
import derl


def get_simple_parser(add_env_id=True, add_logdir=True, nlogs=1e5):
  """ Creates and returns a simple parser. """
  parser = ArgumentParser()

  def maybe_add(add, *args, **kwargs):
    if add:
      parser.add_argument(*args, **kwargs)

  maybe_add(add_env_id, "--env-id", required=True)
  maybe_add(add_logdir, "--logdir", required=True)
  maybe_add(add_logdir, "--nlogs", type=float, default=nlogs)
  return parser


def get_factories():
  """ Returns factory name to factory class mapping. """
  return {
      k[:-len("Factory")].lower().replace("_", "-"): getattr(derl, k)
      for k in dir(derl) if k != "Factory" and k.endswith("Factory")
  }


def get_env_type(env_id):
  """ Returns the type of environment. """
  env_id = ''.join(env_id.split('-')[:-1])
  if env_id.endswith("NoFrameskip"):
    env_id = env_id[:-len("NoFrameskip")]
  for key, envs in derl.env.list_envs().items():
    if env_id in envs:
      return key
  raise ValueError(f"unknown env_type for {env_id=}")


def main():
  """ Script entry point. """
  parser = get_simple_parser()
  factories = get_factories()
  parser.add_argument("factory", choices=list(factories.keys()))
  args, unknown_args = parser.parse_known_args()
  factory_class = factories[args.factory]

  config = derl.factory.Config.make_for_factory(
      factory_class, get_env_type(args.env_id), unknown_args)
  derl.summary.make_writer(args.logdir)
  factory = factory_class(config)
  env = derl.env.make(args.env_id, **factory.make_env_kwargs(args.env_id))
  alg = factory.make(env, nlogs=args.nlogs)
  alg.learn()


if __name__ == "__main__":
  main()
