import numpy as np
import gym
#env = gym.make('GridWorld-v0')

#env = gym.make('MsPacman-v0')
#env = gym.make('Breakout-v0')
env = gym.make('SpaceInvaders-v0')
#env = gym.make('Go19x19-v0')
#env.reset()
#env.render()


#env = gym.make('FrozenLake-v0')
print(gym.__version__)
#env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
env.reset()
while True:
    env.render()
