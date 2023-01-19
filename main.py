import numpy as np
import gym
from stable_baselines3 import DQN

env = gym.make("memory_maze:MemoryMaze-custom-HD-v0")

obs = env.reset()
actions = [0,1,2]
for action in actions:
    obs, rewards, dones, info = env.step(action)