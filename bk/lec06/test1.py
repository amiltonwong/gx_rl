import numpy as np
import gym
#env = gym.make('GridWorld-v0')
print(gym.__version__)
#env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
env.reset()
while True:
    env.render()