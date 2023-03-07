import numpy as np
import gym
import random

env = gym.make("memory_maze:MemoryMaze-custom-v0")

obs = env.reset()
env.render()

#actions = [0,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
actions = [np.array([-1.0,0.0]), np.array([0.0,-1.0])]
for action in range(10):
    obs, rewards, dones, info = env.step(random.choice(actions))
    print(obs)
    env.render()