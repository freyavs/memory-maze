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
        self.smell_range = 3 
        self.action_space = _convert_to_space(env.action_spec())
        self.observation_space = _convert_to_space(self.observation_spec())
        self.state = None
    
    def action_spec(self):
        return self.env.action_spec()

    def observation_spec(self):
        spec = self.env.observation_spec()
        smell_spec = specs.BoundedArray((1,), np.int64, minimum=0, maximum=self.smell_range)
        touch_spec = specs.BoundedArray((3,), np.int64, minimum=0, maximum=1)
        spec = {'image': spec['image'], 'smell': smell_spec, 'touch': touch_spec, 'agent_pos': spec["agent_pos"], 'target_pos': spec["target_pos"], 'maze_layout': spec["maze_layout"]}
        return spec

    def reset(self) -> Any:
        ts = self.env.reset()
        self.state = ts.observation
        obs = self._transform_observation(ts)
        obs["agent_pos"] = ts.observation["agent_pos"]
        obs["target_pos"] = ts.observation["target_pos"]
        obs["maze_layout"] = ts.observation["maze_layout"]
        return obs
    
    def render(self, mode = "human"):
        if mode == "human":
            rgb_state = cv2.cvtColor(self.state['image'], cv2.COLOR_BGR2RGB)
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

        obs = self._transform_observation(ts)
        obs["agent_pos"] = ts.observation["agent_pos"]
        obs["target_pos"] = ts.observation["target_pos"]
        obs["maze_layout"] = ts.observation["maze_layout"]

        return obs, ts.reward, done, info

    def _transform_observation(self, ts):
        smell = np.array([self._calculate_smell(ts)])
        touch = np.array(self._calculate_touch(ts))
        observation = {"image": ts.observation["image"], "smell": smell, "touch": touch}
        return observation
    
    def _calculate_smell(self, ts):
        if ts.reward is None:
            smell_value = 0
        else:
            distance = abs(int(np.round(ts.reward))) # else use target_pos and agent_pos
            smell_range = self.smell_range 
            smell_value = 0
            if distance < smell_range:
                smell_value = smell_range - distance

        return smell_value
    
    def _calculate_touch(self, ts):
        agent_dir = ts.observation["agent_dir"]
        agent_pos = ts.observation["agent_pos"]
        maze_layout = ts.observation["maze_layout"]

        agent_main_dir = np.argmax(np.abs(agent_dir))
        agent_dir[1-agent_main_dir] = 0
        agent_dir = np.rint(agent_dir)

        agent_pos = (agent_pos[0]-0.5, agent_pos[1]-0.5)
        agent_pos = np.clip(np.rint(agent_pos).astype(int), 0, maze_layout.shape[0]-1)

        dir_x, dir_y = agent_dir 
        agent_x, agent_y = agent_pos

        left = (0, dir_x) if dir_x else (-dir_y, 0)
        right = (0, -dir_x) if dir_x else (dir_y, 0)

        forward_pos_x, forward_pos_y = (agent_pos + agent_dir).astype(int)
        left_pos_x, left_pos_y = (agent_pos + left).astype(int)
        right_pos_x, right_pos_y = (agent_pos + right).astype(int)

        # debugging
        #maze_layout = np.ones(maze_layout.shape)
        #maze_layout[agent_y][agent_x] = 2 
        #print(maze_layout)

        if forward_pos_x < 0 or forward_pos_y < 0 or forward_pos_x >= maze_layout.shape[0] or forward_pos_y >= maze_layout.shape[1]:
            wall_forward = 1 
        else:
            wall_forward = 1 - maze_layout[forward_pos_y][forward_pos_x]

        if left_pos_x < 0 or left_pos_y < 0 or left_pos_x >= maze_layout.shape[0] or left_pos_y >= maze_layout.shape[1]:
            wall_left = 1 
        else:
            wall_left = 1 - maze_layout[left_pos_y][left_pos_x]

        if right_pos_x < 0 or right_pos_y < 0 or right_pos_x >= maze_layout.shape[0] or right_pos_y >= maze_layout.shape[1]:
            wall_right = 1
        else:
            wall_right = 1 - maze_layout[right_pos_y][right_pos_x]
        
        return [wall_forward,wall_left,wall_right]


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
