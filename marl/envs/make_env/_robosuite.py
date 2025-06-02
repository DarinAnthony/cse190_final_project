from typing import Any, Dict, List, Callable

import robosuite
from marl.envs.wrappers._robosuite import RecordVideoWrapper

def is_robosuite_env(env_id: str):
    """
    Check if the environment is a robosuite environment
    """
    try:
        import robosuite
    except ImportError:
        return False
    from robosuite.environments import ALL_ENVIRONMENTS

    return env_id in ALL_ENVIRONMENTS

def env_factory(
    env_id: str, 
    idx: int, 
    record_video_path: str = None, 
    record_video_interval: int = 2000,
    env_kwargs: Dict[str, Any] = {}, 
    wrappers: List[Callable] = [],
    **kwargs: Any
    ):
    """
    Creates a factory function that initializes and returns a wrapped robosuite environment for Gymnasium compatibility.

    Args:
        env_id (str): ID or name of the robosuite environment to create.
        idx (int): Index of the environment, used to control video recording.
        record_video_path (str, optional): Path to save episode recordings. Recording is only enabled if provided and `idx == 0`.
        record_video_interval (int, optional): Interval between video recordings (in steps).
        env_kwargs (Dict[str, Any]): Additional arguments passed to `robosuite.make`.
        wrappers (List[Callable]): A list of wrapper functions to apply to the environment.
        **kwargs (Any): Additional keyword arguments (currently unused, but included for extensibility).

    Returns:
        A function that when called, returns the fully initialized and wrapped environment.
    """
    def _init():
        env = robosuite.make(
            env_id,
            **env_kwargs
        )
        # env = RobosuiteWrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideoWrapper(env, record_video_path, step_trigger=lambda x: x % record_video_interval == 0)
        return env
    return _init

  