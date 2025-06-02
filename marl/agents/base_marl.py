from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from marl.policies.base_policy import BasePolicy
from marl.algorithms.base import BaseAlgorithm


class BaseMARLAgent(ABC):
    """
    Abstract base class for Multi-Agent Reinforcement Learning (MARL) agents.

    This class provides a structured setup where multiple agents 
    act in a shared environment. It encapsulates:
    
    - Initialization of specified actor/critic observation keys
    - Buffer creation via the algorithm class
    - Observation normalization 
    - Processing of observations into tensors for actor/critic models
    - Post-processing of action dictionaries into a single tensor for environment input


    Args:
        env: The environment object, which must have `num_envs` property for buffer setup.
        policy: The policy object managing action selection.
        algorithm: The learning algorithm defining buffer usage and updates.
        observation_config: Keys and dimensions for actor/critic observations.
        num_transitions_per_env: Rollout length used for buffer allocation.
        normalize_observations: Whether to normalize inputs using running statistics.
        logger: Optional logging utility.
        device: Device on which tensors and models should be placed.
    """

    def __init__(
        self, 
        env: Any,
        policy: BasePolicy, 
        algorithm: BaseAlgorithm,
        observation_config: Dict[str, List[str]],
        num_transitions_per_env: int,
        normalize_observations: bool,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.policy = policy
        self.algorithm = algorithm
        self.logger = logger
        self.device = device

        self.observation_config = observation_config
        self.num_transitions_per_env = num_transitions_per_env
        self.normalize_observations = normalize_observations

        self.actor_obs_keys = observation_config['actor_obs_keys']
        self.critic_obs_keys = observation_config.get('critic_obs_keys') or self.actor_obs_keys

        self._post_init()

    def _post_init(self):
        """
        Post-initialization step to compute total observation dimensions per agent (for normalization) and
        initialize agent/critic lists, buffers, and normalizers.
        """
        self.num_actor_obs = {
            agent: sum(dim for _, dim in keys)
            for agent, keys in self.actor_obs_keys.items()
        }

        self.num_critic_obs = {
            critic: sum(dim for _, dim in keys)
            for critic, keys in self.critic_obs_keys.items()
        }

        self.actors = list(self.actor_obs_keys.keys())
        self.critics = list(self.critic_obs_keys.keys())

        self._init_buffers()
        self._init_normalizers()

    def _init_buffers(self):
        """
        Initializes storage buffers for rollouts using the provided algorithm.
        """
        self.buffers = self.algorithm._init_storage(self.env.num_envs, self.num_transitions_per_env)

    def _init_normalizers(self):
        """
        Initializes input normalizers for actor and critic observations if enabled.
        """
        if self.normalize_observations:
            from marl.networks import EmpiricalNormalization
            self.actor_obs_normalizer = {
                agent: EmpiricalNormalization(shape=[self.num_actor_obs[agent]], until=1e8).to(self.device)
                for agent in self.actors
            }
            self.critic_obs_normalizer = {
                critic: EmpiricalNormalization(shape=[self.num_critic_obs[critic]], until=1e8).to(self.device)
                for critic in self.critics
            }
        else:
            self.actor_obs_normalizer = {agent: torch.nn.Identity() for agent in self.actors}
            self.critic_obs_normalizer = {critic: torch.nn.Identity() for critic in self.critics}

    def process_observations(
        self, 
        all_obs: Dict[str, np.ndarray]
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Constructs actor and critic observation tensors from raw observation dictionary.

        Args:
            all_obs: Dictionary mapping observation names to numpy arrays.

        Returns:
            Tuple containing:
                - actor_obs: Concatenated tensors per agent for policy input.
                - critic_obs: Concatenated tensors per critic for value function input.
        """
        actor_obs = {
            agent: torch.from_numpy(np.concatenate([all_obs[key] 
                                                    for key, _ in self.actor_obs_keys[agent]], axis=-1)
                                                    ).float().to(self.device)
            for agent in self.actors
        }
        critic_obs = {
            critic: torch.from_numpy(np.concatenate([all_obs[key] 
                                                     for key, _ in self.critic_obs_keys[critic]], axis=-1)
                                                     ).float().to(self.device)
            for critic in self.critics
        }
        return actor_obs, critic_obs

    def _normalize_observations(
        self, 
        actor_obs: Dict[str, torch.Tensor], 
        critic_obs: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Applies observation normalization to actor and critic observations.

        Args:
            actor_obs: Actor observation tensors.
            critic_obs: Critic observation tensors.

        Returns:
            Tuple containing normalized actor and critic observation dictionaries.
        """
        actor_obs = {agent: self.actor_obs_normalizer[agent](obs) for agent, obs in actor_obs.items()}
        critic_obs = {critic: self.critic_obs_normalizer[critic](obs) for critic, obs in critic_obs.items()}
        return actor_obs, critic_obs

    def process_actions(self, actions: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Converts dictionary of actions into a single concatenated tensor for the environment.

        Args:
            actions: Dictionary mapping agent to actions.

        Returns:
            Concatenated actions ready to be passed to the environment.
        """
        return np.concatenate([actions[agent].cpu() for agent in self.actors], axis=-1)

    def train_mode(self):
        """
        Sets the policy and normalizers to training mode.
        """
        self.policy.train()
        if self.normalize_observations:
            for norm in self.actor_obs_normalizer.values():
                norm.train()
            for norm in self.critic_obs_normalizer.values():
                norm.train()

    def eval_mode(self):
        """
        Sets the policy and normalizers to evaluation mode.
        """
        self.policy.eval()
        if self.normalize_observations:
            for norm in self.actor_obs_normalizer.values():
                norm.eval()
            for norm in self.critic_obs_normalizer.values():
                norm.eval()
    
    def act_inference(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Acts for inference

        args:
            obs: Dictionary mapping observation names to numpy arrays. Same format during training. 

        Returns:
            numpy array of actions ready for the environment
        """
        self.eval_mode()
        actor_obs, critic_obs = self.process_observations(obs)
        actor_obs, critic_obs = self._normalize_observations(actor_obs, critic_obs)
        actions = {}
        for agent in self.actors:
            actions[agent] = self.policy.act(actor_obs[agent], agent_id=agent)
        actions = self.process_actions(actions)
        return actions

    def save(self, path: str):
        """
        Save the policy and normalizers

        args:
            path: Path to save the policy and normalizers
        """
        self.policy.save(path)
        for agent in self.actors:
            self.actor_obs_normalizer[agent].save(path + f"/{agent}_actor_obs_normalizer.pt")
        for critic in self.critics:
            self.critic_obs_normalizer[critic].save(path + f"/{critic}_critic_obs_normalizer.pt")


    def load(self, path: str):
        """
        Load the policy and normalizers

        args:
            path: Path to load the policy and normalizers
        """
        self.policy.load(path)
        for agent in self.actors:
            self.actor_obs_normalizer[agent].load(path + f"/{agent}_actor_obs_normalizer.pt")
        for critic in self.critics:
            self.critic_obs_normalizer[critic].load(path + f"/{critic}_critic_obs_normalizer.pt")

    @abstractmethod
    def learn(self, total_iterations: int = 1000) -> None:
        """Abstract method that subclasses must implement to train the agent."""
        ...
    