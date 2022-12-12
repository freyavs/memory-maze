import numpy as np
import gym


env = gym.make("memory_maze:MemoryMaze-custom-v0")

# First reset the env to get the initial observation
obs = env.reset()
#_ = vis_image(obs, show=True)

# Next do three steps and plot the observation (by calling  env.step(action) )
for i in range(3):
    obs, reward, terminal, _ = env.step(3)
    #_ = vis_image(obs, show=True)