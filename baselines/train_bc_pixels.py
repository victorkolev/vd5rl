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

import flax
from jaxrl2.agents import PixelIQLLearner
from jaxrl2.data import Dataset
from jaxrl2.evaluation import evaluate

import d4rl2

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'widowx-stitch-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/bc_pixels_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


class JAXRL2Pixels(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space
        space = gym.spaces.Box(low=old_space.low[..., np.newaxis],
                               high=old_space.high[..., np.newaxis],
                               dtype=old_space.dtype)
        self.observation_space = gym.spaces.Dict(dict(pixels=space))

    def observation(self, observation):
        return {'pixels': observation[..., np.newaxis]}


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb'))

    eval_env = gym.make(FLAGS.env_name)
    eval_env = JAXRL2Pixels(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps
    agent = PixelIQLLearner(FLAGS.seed, eval_env.observation_space.sample(),
                            eval_env.action_space.sample(), **kwargs)

    dataset = eval_env.q_learning_dataset()

    replay_buffer = Dataset(dataset_dict=dataset)

    replay_buffer.seed(FLAGS.seed)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = replay_buffer.sample(FLAGS.batch_size)
        batch = batch.unfreeze()
        obs = np.stack([batch['observations'], batch['next_observations']],
                       axis=-1)
        batch['observations'] = {'pixels': obs}
        batch['next_observations'] = {}
        batch['dones'] = batch['trajectory_ends']
        batch['masks'] = 1 - batch['terminals'].astype(np.float32)
        batch = flax.core.frozen_dict.FrozenDict(batch)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v, i)
                else:
                    summary_writer.histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                summary_writer.scalar(f'evaluation/{k}', v, i)


if __name__ == '__main__':
    app.run(main)
