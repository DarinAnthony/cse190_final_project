from typing import Any, Dict, List, Callable

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
from marl.envs.wrappers._gymasnium import GymnasiumWrapper

def is_gymnasium_env(env_id: str):
    """
    Check if the environment is a gymnasium environment
    """
    try:
        import gymnasium as gym
    except ImportError:
        return False
    from gymnasium.envs.registration import registry
  
    return env_id in registry

def env_factory(
    env_id: str, 
    idx: int, 
    max_episode_steps: int = 200, 
    record_video_path: str = None, 
    record_video_interval: int = 2000,
    env_kwargs: Dict[str, Any] = {}, 
    wrappers: List[Callable] = []):
    """
    Creates a factory function that initializes and returns a wrapped Gymnasium environment.


    Args:
        env_id (str): ID of the Gymnasium environment to create.
        idx (int): Index of the environment, used for vector envs.
        max_episode_steps (int): Maximum number of steps per episode before termination.
        record_video_path (str, optional): Directory path to store recorded videos.
        record_video_interval (int, optional): Interval between video recordings (in steps).
        env_kwargs (Dict[str, Any]): Additional keyword arguments passed to the environment constructor.
        wrappers (List[Callable]): A list of callable wrappers to apply to the environment.

    Returns:
       A function that when called, returns the fully wrapped Gymnasium environment.
    """
    def _init():
        env = gym.make(
            env_id,
            **env_kwargs
        )
        env = GymnasiumWrapper(env) #Codebase expects obs to be a dict
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideo(env, record_video_path, episode_trigger=lambda x: x % record_video_interval == 0)
        return env
    return _init