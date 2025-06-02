import gymnasium as gym
import imageio
from typing import Callable

class EpisodeStatsWrapper(gym.Wrapper):
    """
    Adds additional info. Anything that goes in the stats wrapper is logged to tensorboard/wandb under train_stats and test_stats
    """

    def reset(self, *, seed=None, options=None):
        self.eps_seed = seed
        obs, info = super().reset(seed=seed, options=options)
        self.eps_ret = 0
        self.eps_len = 0
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["seed"] = self.eps_seed
        if "episode" in info:
            info["eps_ret"] = info["stats"]["return"]
            info["eps_len"] = info["stats"]["episode_len"]
        else:
            self.eps_ret += reward
            self.eps_len += 1
            info["eps_ret"] = self.eps_ret
            info["eps_len"] = self.eps_len
        return observation, reward, terminated, truncated, info
