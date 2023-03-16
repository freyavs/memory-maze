from typing import Any, Tuple
import numpy as np

import dm_env
import gym
from dm_env import specs
from gym import spaces
import cv2


class GymWrapper(gym.Env):

    def __init__(self, env: dm_env.Environment):
        self.env = env
        self.action_space = _convert_to_space(env.action_spec())
        self.observation_space = _convert_to_space(env.observation_spec())
        self.state = None
    
    def action_spec(self):
        return self.env.action_spec()

    def observation_spec(self):
        return self.env.observation_spec()

    def reset(self) -> Any:
        ts = self.env.reset()
        self.state = ts.observation
        return ts.observation
    
    def render(self, mode = "human"):
        if mode == "human":
            rgb_state = cv2.cvtColor(self.state, cv2.COLOR_BGR2RGB)
            cv2.imshow("Environment state", rgb_state)
            cv2.waitKey(0)
        else:
            print(self.state)

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        ts = self.env.step(action)
        self.state = ts.observation
        assert not ts.first(), "dm_env.step() caused reset, reward will be undefined."
        assert ts.reward is not None
        done = ts.last() or ts.reward == 5 # > 100# reset when time is up or target is reached
        terminal = ts.last() and ts.discount == 0.0
        info = {}
        if done and not terminal:
            info['TimeLimit.truncated'] = True  # acme.GymWrapper understands this and converts back to dm_env.truncation()
        info["last"] = ts.last()
        info["first"] = ts.first()
        info["discount"] = ts.discount
        info["done"] = done
        info["terminal"] = terminal
        info["target_reached"] = ts.reward == 5 #> 100
        return ts.observation, ts.reward, done, info


def _convert_to_space(spec: Any) -> gym.Space:
    # Inverse of acme.gym_wrappers._convert_to_spec

    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)

    if isinstance(spec, specs.BoundedArray):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=spec.minimum.item() if len(spec.minimum.shape) == 0 else spec.minimum,
            high=spec.maximum.item() if len(spec.maximum.shape) == 0 else spec.maximum)
    
    if isinstance(spec, specs.Array):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=-np.inf,
            high=np.inf)

    if isinstance(spec, tuple):
        return spaces.Tuple(_convert_to_space(s) for s in spec)

    if isinstance(spec, dict):
        return spaces.Dict({key: _convert_to_space(value) for key, value in spec.items()})

    raise ValueError(f'Unexpected spec: {spec}')
