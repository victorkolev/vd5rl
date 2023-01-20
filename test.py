import gym

import d4rl2.envs.a1

env = gym.make('a1-umaze-diverse-v0')
dataset = env.q_learning_dataset()
batch = dataset.sample(256)

print(batch)