import numpy as np
import gym
from stable_baselines3 import DQN

env = gym.make("memory_maze:MemoryMaze-custom-v0")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)

# saving and loading demonstration
#model.save("deepq_maze")
#del model # remove to demonstrate saving and loading
#model = DQN.load("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()