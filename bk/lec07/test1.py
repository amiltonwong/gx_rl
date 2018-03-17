import gym
import numpy as np
import matplotlib.pyplot as plt

#创建FrozenLake的仿真环境
env = gym.make('FrozenLake-v0')
state = env.reset()
print(env.action_space)