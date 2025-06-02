from typing import Callable, Optional, Any
from functools import partial

import marl.envs.make_env._robosuite as _robosuite
import marl.envs.make_env._gymnasium as _gymnasium
from marl.envs.wrappers.common import EpisodeStatsWrapper
from marl.envs.wrappers._robosuite import GymWrapper

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

def wrap_env(env, wrappers=[]):
    """ Wrap the environment with the given wrappers. """
    for wrapper in wrappers:
        env = wrapper(env)
    return env

def make_env(
    env_id: str,
    env_type: str,
    num_envs: Optional[int] = 1,
    seed: Optional[int] = 0,
    record_video_path: Optional[str] = None,
    record_video_interval: Optional[int] = 2000,
    env_kwargs: dict = dict(),
    wrappers: list[Callable] = [],
    **kwargs: Any
):
    """
    Creates and returns a vectorized or single-instance environment, wrapped and optionally configured 
    for video recording and statistic tracking.

    This currently supports both `gymnasium` and `robosuite` environments. 


    Args:
        env_id (str): The environment ID (used by gymnasium or robosuite).
        env_type (str): Type of the environment, e.g., "gym:cpu". JAX environments are not yet supported.
        num_envs (int, optional): Number of parallel environments to create. Defaults to 1.
        seed (int, optional): Random seed used to initialize the environment. Defaults to 0.
        record_video_path (str, optional): Directory path to save episode recordings.
        record_video_interval (int, optional): Interval between video recordings (in steps).
        env_kwargs (dict): Additional arguments to pass to the environment constructor.
        wrappers (list[Callable]): List of additional wrappers to apply to each environment.
        **kwargs (Any): Additional arguments passed to the environment factory.

    Returns:
       A (vectorized) Gym-compatible environment instance ready for training or evaluation.
    """
    if env_type == "jax":
        raise NotImplementedError("Jax environment is not implemented yet")

    else:
        context = "fork"
        wrappers = [EpisodeStatsWrapper, *wrappers]


        if _robosuite.is_robosuite_env(env_id):
            env_factory = _robosuite.env_factory
            wrappers = [partial(GymWrapper, flatten_obs=False), *wrappers]
            context = "forkserver"
        elif _gymnasium.is_gymnasium_env(env_id):
            env_factory = _gymnasium.env_factory
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        if env_type == "gym:cpu":
            # create a vector env parallelized across CPUs
            vector_env_cls = partial(AsyncVectorEnv, context=context)
            if num_envs == 1:
                vector_env_cls = SyncVectorEnv
            env = vector_env_cls(
                [
                    env_factory(
                        env_id=env_id,
                        idx=idx,
                        env_kwargs=env_kwargs,
                        record_video_path=record_video_path,
                        record_video_interval=record_video_interval,
                        wrappers=wrappers,
                        **kwargs
                    )
                    for idx in range(num_envs)
                ]
            )
        else:
            env = env_factory(
                env_id=env_id,
                idx=0,
                env_kwargs=env_kwargs,
                record_video_path=record_video_path,
                record_video_interval=record_video_interval,
                wrappers=wrappers,
                **kwargs
            )()
            raise ValueError(f"Unknown environment type: {env_type}")
        env.reset(seed=seed)

        return env
