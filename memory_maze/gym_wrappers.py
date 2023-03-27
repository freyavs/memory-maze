from typing import Any, Tuple
import numpy as np

import dm_env
import gym
from dm_env import specs
from gym import spaces
import cv2
import math

class GymWrapper(gym.Env):

    def __init__(self, env: dm_env.Environment):
        self.env = env
        self.action_space = _convert_to_space(env.action_spec())
        self.observation_space = _convert_to_space(env.observation_spec())

    def reset(self) -> Any:
        ts = self.env.reset()
        return ts.observation

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        ts = self.env.step(action)
        assert not ts.first(), "dm_env.step() caused reset, reward will be undefined."
        assert ts.reward is not None
        done = ts.last()
        terminal = ts.last() and ts.discount == 0.0
        info = {}
        if done and not terminal:
            info['TimeLimit.truncated'] = True  # acme.GymWrapper understands this and converts back to dm_env.truncation()
        return ts.observation, ts.reward, done, info


class GymDreamerWrapper(gym.Env):

    def __init__(self, env: dm_env.Environment):
        self.env = env
        self.action_space = _convert_to_space(env.action_spec())
        self.observation_space = _convert_to_space(self.observation_spec())
        self.state = None
    
    def action_spec(self):
        return self.env.action_spec()

    def observation_spec(self):
        spec = self.env.observation_spec()
        return {'image': spec['image'], 'smell': specs.BoundedArray((1,), np.uint8, minimum=0, maximum=3)}

    def reset(self) -> Any:
        ts = self.env.reset()
        self.state = ts.observation
        return self._transform_observation(ts)
    
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
        done = ts.last() or ts.reward == 0 # reset when time is up or target is reached
        terminal = ts.last() and ts.discount == 0.0
        info = {}
        if done and not terminal:
            info['TimeLimit.truncated'] = True  # acme.GymWrapper understands this and converts back to dm_env.truncation()
        
        # set extra parameters for dreamer
        info["last"] = ts.last()
        info["first"] = ts.first()
        info["discount"] = ts.discount
        info["done"] = done
        info["terminal"] = terminal
        info["target_reached"] = ts.reward == 0 

        return self._transform_observation(ts), ts.reward, done, info

    def _transform_observation(self, ts):
        distance = abs(int(ts.reward or 4)) # else use target_pos and agent_pos
        smell_range = 3 
        smell_value = 0
        if distance < smell_range:
            smell_value = smell_range - distance

        observation = {"image": ts.observation["image"], "smell": [smell_value]}
        return observation

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
