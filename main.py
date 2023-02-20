import numpy as np
import gym

env = gym.make("memory_maze:MemoryMaze-custom-HD-v0")

obs = env.reset()
env.render()

actions = [0,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
for action in actions:
    obs, rewards, dones, info = env.step(action)
    env.render()