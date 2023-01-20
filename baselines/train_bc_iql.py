#! /usr/bin/env python
import os

import gym
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

tf.config.experimental.set_visible_devices([], "GPU")
from jaxrl2.agents import BCLearner, IQLLearner
from jaxrl2.data import Dataset
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym

import d4rl2

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'a1-umaze-diverse-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    os.path.join(os.path.dirname(__file__), 'configs', 'bc_iql_default.py:bc'),
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb'))

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env)
    env.seed(FLAGS.seed)

    dataset = env.q_learning_dataset()
    lim = 1 - 1e-6
    dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
    dataset = Dataset(
        dict(observations=dataset['observations'],
             actions=dataset['actions'],
             next_observations=dataset['next_observations'],
             rewards=dataset['rewards'],
             dones=dataset['trajectory_ends'],
             masks=1 - dataset['terminals'].astype(np.float32)))
    dataset.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay'):
        kwargs['decay_steps'] = FLAGS.max_steps

    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
        **kwargs)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v, i)
                else:
                    summary_writer.histogram(f'training/{k}', v, i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)

            for k, v in eval_info.items():
                summary_writer.scalar(f'evaluation/{k}', v, i)


if __name__ == '__main__':
    app.run(main)
