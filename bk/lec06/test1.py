import numpy as np
import gym
env = gym.make('GridWorld-v0')
env.reset()
while True:
    env.render()