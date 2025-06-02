"""
MARL Component Builders

This module provides functions to build MARL components (environments, policies, 
algorithms, and agents) from configuration files.
"""

from typing import Dict, Any, Tuple

from omegaconf import DictConfig, OmegaConf

from marl.agents.base_marl import BaseMARLAgent
from marl.agents.basic_marl_agent import BasicMARLAgent
from marl.envs.make_env.make_env import make_env
from marl.policies import MultiAgentPolicyBuilder, BasePolicy
from marl.utils.utils import resolve_controller
from marl.algorithms.ppo import PPO
from marl.algorithms.mappo import MAPPO
from marl.algorithms.base import BaseAlgorithm

# =============================================================================
# Environment Builder
# =============================================================================

def instantiate_env(config: DictConfig):
    """
    Build environment from configuration.
    
    Args:
        config: Environment configuration containing:
            - id: Environment identifier
            - type: Environment type
            - env_kwargs: Environment-specific arguments including controller_configs
    
    Returns:
        Constructed environment instance
    """
    env_config = config.get("environment", config)
    env_kwargs = env_config.get("env_kwargs", {})
    
    # Resolve controller configuration if present
    if "controller_configs" in env_kwargs:
        env_kwargs["controller_configs"] = resolve_controller(env_kwargs["controller_configs"])
    
    return make_env(
        env_id=env_config.get("id"),
        env_type=env_config.get("type"),
        num_envs=env_config.get("num_envs"),
        seed=env_config.get("seed"),
        max_episode_steps=env_config.get("max_episode_steps", 200),
        record_video_path=config.get("video_path") if config.get("video") else None,
        record_video_interval=config.get("video_interval") if config.get("video") else None,
        env_kwargs=env_kwargs
    )


# =============================================================================
# Policy Builder
# =============================================================================

def instantiate_policy(config: DictConfig):
    """
    Build multi-agent policy from configuration.
    
    Args:
        config: Policy configuration containing:
            - components: Dictionary of component configurations
            - connections: Optional list of component connections
    
    Returns:
        Constructed multi-agent policy instance
    """
    builder = MultiAgentPolicyBuilder()
        
    # Add components from config
    for component_id, component_config in config["components"].items():
        builder.add_component(
            component_id=component_id,
            **component_config
        )
    
    # Add connections from config if specified
    if "connections" in config:
        for connection in config["connections"]:
            builder.add_connection(**connection)
    
    return builder.build()


# =============================================================================
# Algorithm Builder
# =============================================================================

def instantiate_algorithm(algorithm_config: Dict[str, Any], policy: MultiAgentPolicyBuilder):
    """
    Build algorithm from configuration.
    
    Args:
        policy: Multi-agent policy instance
        config: Algorithm configuration parameters
    
    Returns:
        Constructed algorithm instance (currently PPO)
    """
    algorithm_kwargs = algorithm_config.get("kwargs", {})
    if algorithm_config.get("name") == "PPO":
        agent_hyperparams = parse_agent_configs(algorithm_config)
        normalize_advantage_per_mini_batch = algorithm_kwargs.get("normalize_advantage_per_mini_batch", False)

        return PPO(policy, agent_hyperparams, normalize_advantage_per_mini_batch)
    elif algorithm_config.get("name") == "MAPPO":
        agent_hyperparams = parse_agent_configs(algorithm_config)
        normalize_advantage_per_mini_batch = algorithm_kwargs.get("normalize_advantage_per_mini_batch", False)
        actor_critic_mapping = algorithm_config.get("agent_mapping", None)
        return MAPPO(policy, agent_hyperparams, normalize_advantage_per_mini_batch, actor_critic_mapping)
    else:
        raise ValueError(f"Algorithm {algorithm_config.get('name')} not supported")


def parse_agent_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Generate individual agent configurations by merging global and agent-specific parameters.
    
    Args:
        config: Algorithm configuration containing:
            - global: Global parameters applied to all agents
            - agent_specific_hyperparams: Agent-specific parameter overrides
            - actors: List of actor names
            - critics: List of critic names
    
    Returns:
        Dictionary with structure:
        {
            'actors': {agent_id: {params}},
            'critics': {agent_id: {'learning_rate': value}}
        }
    """
    global_params = config['global']
    agent_specific = config.get('agent_specific_hyperparams', {})
    actors = config['actors']
    critics = config['critics']
    
    agent_configs = {
        'actors': {},
        'critics': {}
    }
    
    for agent_name in actors:
        # Start with global parameters for actor
        agent_config = global_params.copy()
        
        # Apply actor-specific parameters
        if agent_name in agent_specific:
            for key, value in agent_specific[agent_name].items():
                if key in agent_config:
                    agent_config[key] = value
        
        agent_configs['actors'][agent_name] = agent_config
    
    for agent_name in critics:
        agent_config = {}
        if agent_name in agent_specific: 
            agent_config["learning_rate"] = agent_specific[agent_name]["learning_rate"]
            agent_config["max_grad_norm"] = agent_specific[agent_name]["max_grad_norm"]
        else:
            agent_config["learning_rate"] = global_params["learning_rate"]
            agent_config["max_grad_norm"] = global_params["max_grad_norm"]
        agent_configs['critics'][agent_name] = agent_config
    
    return agent_configs


# =============================================================================
# Observation Configuration Helpers
# =============================================================================

def _extract_observation_keys(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract actor and critic observation keys from agent configuration.
    
    Args:
        config: Agent configuration containing observation specifications
    
    Returns:
        Tuple of (actor_obs_keys, critic_obs_keys) dictionaries
    """
    actor_obs_keys = {
        agent: config[agent]["actor_observations"] 
        for agent in config 
        if agent != "kwargs" and agent != "agent_class" and "actor_observations" in config[agent]
    }
    
    critic_obs_keys = {
        agent: config[agent]["critic_observations"]
        for agent in config 
        if agent != "kwargs" and agent != "agent_class" and "critic_observations" in config[agent]
    }
    
    return actor_obs_keys, critic_obs_keys


def _extract_agent_kwargs(config: DictConfig) -> bool:
    """
    Extract agent keyword arguments from configuration.
    
    Args:
        config: Configuration containing agent kwargs
    
    Returns:
        Tuple of (normalize_observations, preprocess_observations) flags
    """
    agent_kwargs = config["kwargs"]
    normalize_observations = agent_kwargs["normalize_observations"]
    
    return normalize_observations


# =============================================================================
# Agent Builder 
# =============================================================================

def instantiate_agent(agent_config: DictConfig, env: Any, policy: MultiAgentPolicyBuilder, algorithm: BaseAlgorithm) -> BaseMARLAgent:
    """
    Build agent from configuration.
    """
    actor_obs_keys, critic_obs_keys = _extract_observation_keys(agent_config)
    
    # Extract agent behavioral flags
    normalize_observations = _extract_agent_kwargs(agent_config)
    
    # Create unified observation configuration
    observation_config = {
        "actor_obs_keys": actor_obs_keys,
        "critic_obs_keys": critic_obs_keys
    }
    num_transitions_per_env = agent_config.get("num_transitions_per_env", 200)
    if agent_config.get("agent_class") == "BasicMARLAgent":
        return BasicMARLAgent(
        env=env,
        policy=policy,
        algorithm=algorithm,
        observation_config=observation_config,
        num_transitions_per_env=num_transitions_per_env,
        normalize_observations=normalize_observations,
        device = policy.device
    )
    else:
        raise ValueError(f"Agent class {agent_config.get('agent_class')} not supported")

# =============================================================================
# Main Builder Function
# =============================================================================

def instantiate_all(config: DictConfig) -> Tuple[Any, BasePolicy, BaseAlgorithm, BaseMARLAgent]:
    """
    Build all MARL components from configuration.
    
#     This is the main factory function that constructs the complete MARL pipeline
#     including environment, policy, algorithm, and agent from a unified configuration.
    
#     Args:
#         config: Complete MARL configuration containing:
#             - environment: Environment configuration
#             - policy: Policy configuration  
#             - algorithm: Algorithm configuration
#             - agent: Agent configuration including observation specs
#             - device: Optional device specification
    
    Returns:
        Tuple of (environment, agent) ready for training
    """
    config = OmegaConf.to_container(config, resolve=True)
    env = instantiate_env(config)
    policy = instantiate_policy(config["policy"])
    algorithm = instantiate_algorithm(config["algorithm"], policy)
    agent = instantiate_agent(
        env=env,
        policy=policy,
        algorithm=algorithm,
        agent_config=config["agent"]
    )
    return env, policy, algorithm, agent
    