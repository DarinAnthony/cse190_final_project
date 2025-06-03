from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from marl.storage.rollout_storage import Transition, RolloutStorage
from marl.algorithms.base import BaseAlgorithm

class PPO(BaseAlgorithm):
  """
  Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).
  
  Implementation adapted from https://github.com/leggedrobotics/rsl_rl for multi-agent 
  reinforcement learning scenarios.
  
  This class supports both single-agent and multi-agent training, with per-agent
  hyperparameters and storage management. This class assumes that each agent contains
  both an actor and a critic.

  Args:
      policy: Multi-agent policy network
      agent_hyperparams: Dictionary mapping agent_id to hyperparameter dictionaries
      normalize_advantage_per_mini_batch: Whether to normalize advantages per mini-batch
    
  Raises:
      ValueError: If the number of actors and critics are not the same
  """
  def __init__(
      self,
      policy,
      agent_hyperparams: Dict[str, Dict[str, Any]],
      normalize_advantage_per_mini_batch: bool = False,
      mappo: bool = False,
    ):
    self.policy = policy
    self.optimizers = {}
    self.storage = {}
    self.transitions = {agent_id: Transition() for agent_id in agent_hyperparams['actors']}

    self.actors = [actor_id for actor_id in agent_hyperparams['actors']]
    self.critics = [critic_id for critic_id in agent_hyperparams['critics']]

    if not mappo:
       if len(self.actors) != len(self.critics):
        raise ValueError("Number of actors and critics must be the same for PPO")

    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    self._setup_hyperparams(agent_hyperparams)
    self._setup_optimizers()

  def _setup_hyperparams(self, agent_hyperparams: Dict[str, Dict[str, Any]]) -> None:
    """ Setup hyperparams for each agent """
    actor_hyperparams = agent_hyperparams['actors']
    critic_hyperparams = agent_hyperparams['critics']

    self.clip_param = {agent_id: actor_hyperparams[agent_id]["clip_param"] for agent_id in self.actors}
    self.num_learning_epochs = {agent_id: actor_hyperparams[agent_id]["num_learning_epochs"] for agent_id in self.actors}
    self.num_mini_batches = {agent_id: actor_hyperparams[agent_id]["num_mini_batches"] for agent_id in self.actors}
    self.value_loss_coef = {agent_id: actor_hyperparams[agent_id]["value_loss_coef"] for agent_id in self.actors}
    self.entropy_coef = {agent_id: actor_hyperparams[agent_id]["entropy_coef"] for agent_id in self.actors}
    self.gamma = {agent_id: actor_hyperparams[agent_id]["gamma"] for agent_id in self.actors}
    self.lambda_ = {agent_id: actor_hyperparams[agent_id]["lambda_"] for agent_id in self.actors}
    self.use_clipped_value_loss = {agent_id: actor_hyperparams[agent_id]["use_clipped_value_loss"] for agent_id in self.actors}
    self.desired_kl = {agent_id: actor_hyperparams[agent_id]["desired_kl"] for agent_id in self.actors}
    self.schedule = {agent_id: actor_hyperparams[agent_id]["schedule"] for agent_id in self.actors}
    self.total_timesteps = {agent_id: actor_hyperparams[agent_id]["total_timesteps"] for agent_id in self.actors}

    #Meaningful parameters for different actors and critics
    self.learning_rate = {agent_id: actor_hyperparams[agent_id]["learning_rate"] for agent_id in self.actors}
    self.learning_rate.update({agent_id: critic_hyperparams[agent_id]["learning_rate"] for agent_id in self.critics})

    self.max_grad_norm = {agent_id: actor_hyperparams[agent_id]["max_grad_norm"] for agent_id in self.actors}
    self.max_grad_norm.update({agent_id: critic_hyperparams[agent_id]["max_grad_norm"] for agent_id in self.critics})

  def _setup_optimizers(self):
    """ Setup optimizers for each agent """
    for actor_id, critic_id in zip(self.actors, self.critics):
        # Check if learning rates are the same
        if self.learning_rate[actor_id] == self.learning_rate[critic_id]:
            if actor_id == critic_id: #This is a single actor-critic network
               print(f"SETTING UP OPTIMIZER FOR SINGLE ACTOR-CRITIC NETWORK {actor_id}")
               self.optimizers[actor_id] = optim.Adam(
                self.policy.parameters(agent_id=actor_id),
                lr=self.learning_rate[actor_id]
            )
            else: #separate actor and critic networks
                print(f"SETTING UP OPTIMIZER FOR SEPARATE ACTOR AND CRITIC NETWORKS {actor_id} and {critic_id}")
                # Use single optimizer for both actor and critic
                self.optimizers[actor_id] = optim.Adam(
                    itertools.chain(self.policy.parameters(agent_id=actor_id), self.policy.parameters(agent_id=critic_id)),
                    lr=self.learning_rate[actor_id]
                )
                # Store reference to same optimizer for critic
                self.optimizers[critic_id] = self.optimizers[actor_id]
        else:
            print(f"SETTING UP DIFFERENT OPTIMIZERS FOR SEPARATE ACTOR AND CRITIC NETWORKS {actor_id} and {critic_id}")
            # Use separate optimizers when learning rates differ
            self.optimizers[actor_id] = optim.Adam(
                self.policy.parameters(agent_id=actor_id),
                lr=self.learning_rate[actor_id]
            )
            self.optimizers[critic_id] = optim.Adam(
                self.policy.parameters(agent_id=critic_id),
                lr=self.learning_rate[critic_id]
            )

  def _init_storage(
        self, 
        num_envs: int, 
        num_transitions_per_env: int, 
        ):
    """
    Initialize storage for all agents.
    
    Args:
        num_envs: Number of environments
        num_transitions_per_env: Number of transitions to store per environment (rollout)
    """
    for actor_id, critic_id in zip(self.actors, self.critics):
      self.storage[actor_id] = RolloutStorage(
              num_envs=num_envs,
              num_transitions_per_env=num_transitions_per_env,
              actor_obs_dim=self.policy.components[actor_id].network_kwargs['actor_obs_dim'],
              critic_obs_dim=self.policy.components[critic_id].network_kwargs['critic_obs_dim'],
              action_dim=self.policy.components[actor_id].network_kwargs['num_actions'],
              device=self.policy.device
          )

  def act(self, actor_obs, critic_obs=None):
    """
    Execute actions for all agents.
    
    Args:
        actor_obs: Observations for actor (can be dict mapping agent_id->obs or single obs)
        critic_obs: Observations for critic (can be dict mapping agent_id->obs or single obs)
        agent_id: Specific agent to act for, or None to act for all agents
    
    Returns:
        If agent_id specified: actions tensor for that agent
        If agent_id is None: dict mapping agent_id -> actions tensor
    """
    all_actions = {}
    for actor_id, critic_id in zip(self.actors, self.critics):
      self.transitions[actor_id].actions = self.policy.act(actor_obs[actor_id], agent_id=actor_id).detach()
      self.transitions[actor_id].values = self.policy.evaluate(critic_obs[critic_id], agent_id=critic_id).detach()
      self.transitions[actor_id].actions_log_prob = self.policy.get_actions_log_prob(self.transitions[actor_id].actions, actor_id).detach()
      self.transitions[actor_id].action_mean = self.policy.get_action_mean(actor_id).detach() 
      self.transitions[actor_id].action_sigma = self.policy.get_action_std(actor_id).detach()
      self.transitions[actor_id].actor_observations = actor_obs[actor_id]
      self.transitions[actor_id].critic_observations = critic_obs[critic_id]
      all_actions[actor_id] = self.transitions[actor_id].actions

    return all_actions


  def process_env_step(self, rewards, dones) -> None:
    """
    Process environment step for agents.
    
    Args:
        rewards: Rewards (can be dict mapping agent_id->reward or single reward)
        dones: Done flags (can be dict mapping agent_id->done or single done)
        agent_id: Specific agent to process for, or None to process for all agents
    """
    for actor_id in self.actors:
      if isinstance(rewards, dict): #If we want to split rewards between agents
        self.transitions[actor_id].rewards = torch.from_numpy(rewards[actor_id]).clone()
      else:
        self.transitions[actor_id].rewards = torch.from_numpy(rewards).clone()
      self.transitions[actor_id].dones = torch.from_numpy(dones).clone()

      self.storage[actor_id].add(self.transitions[actor_id])
      self.transitions[actor_id] = Transition()


  def compute_returns(self, last_critic_obs) -> None:
      """
      Compute returns for agents.
      
      Args:
          last_critic_obs: Last critic observations (can be dict mapping agent_id->obs or single obs)
          agent_id: Specific agent to compute returns for, or None to compute for all agents
      """
      for actor_id, critic_id in zip(self.actors, self.critics):
        last_values = self.policy.evaluate(last_critic_obs[critic_id], agent_id=critic_id).detach()
        self.storage[actor_id].compute_returns(
            last_values,
            self.gamma[actor_id],
            self.lambda_[actor_id],
            not self.normalize_advantage_per_mini_batch
        )
            

  def update(self) -> Dict[str, Any]:
    """ Update all agents """
    all_loss_dicts = {}
    for actor_id, critic_id in zip(self.actors, self.critics):
      loss_dict = self._update_single_agent(actor_id, critic_id)
      all_loss_dicts[actor_id] = loss_dict
    return all_loss_dicts

  def _update_single_agent(self, actor_id: str, critic_id: str, update_critic: bool = True) -> Tuple[float, float, float]:
    """Update a single agent's parameters"""
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0

    generator = self.storage[actor_id].mini_batch_generator(
        self.num_mini_batches[actor_id], 
        self.num_learning_epochs[actor_id]
    )

    for (
        actor_obs_batch,
        critic_obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        ) in generator:
      original_batch_size = actor_obs_batch.shape[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
      self.policy.act(actor_obs_batch, agent_id=actor_id)
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch, agent_id=actor_id)

      # Only compute value if we're updating the critic
      if update_critic:
        value_batch = self.policy.evaluate(critic_obs_batch, agent_id=critic_id)

      mu_batch = self.policy.get_action_mean(actor_id)[:original_batch_size]
      sigma_batch = self.policy.get_action_std(actor_id)[:original_batch_size]
      entropy_batch = self.policy.get_entropy(actor_id)[:original_batch_size]

      if self.desired_kl[actor_id] is not None and self.schedule[actor_id] == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
              torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
              + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
              / (2.0 * torch.square(sigma_batch))
              - 0.5,
              axis=-1,
          )
          kl_mean = torch.mean(kl)

          if kl_mean > self.desired_kl[actor_id] * 2.0:
              self.learning_rate[actor_id] = max(1e-5, self.learning_rate[actor_id] / 1.5)
          elif kl_mean < self.desired_kl[actor_id] / 2.0 and kl_mean > 0.0:
              self.learning_rate[actor_id] = min(1e-2, self.learning_rate[actor_id] * 1.5)

          for param_group in self.optimizers[actor_id].param_groups:
            param_group["lr"] = self.learning_rate[actor_id]

      ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param[actor_id], 1.0 + self.clip_param[actor_id]
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Compute value loss only if updating critic
      if update_critic:
        if self.use_clipped_value_loss[actor_id]:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param[actor_id], self.clip_param[actor_id]
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
      else:
        value_loss = torch.tensor(0.0)  # Zero loss when not updating critic

      # Actor loss (always computed)
      loss = surrogate_loss + self.value_loss_coef[actor_id] * value_loss - self.entropy_coef[actor_id] * entropy_batch.mean()
      
      # Handle optimizer updates based on whether critic is being updated
      if self.optimizers[actor_id] is self.optimizers[critic_id]:
        # Same optimizer for both actor and critic
        self.optimizers[actor_id].zero_grad()
          
        loss.backward()
        
        # Clip gradients
        if update_critic:
          if actor_id == critic_id:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(agent_id=actor_id), 
                self.max_grad_norm[actor_id]
            )
          else:
            nn.utils.clip_grad_norm_(
                itertools.chain(self.policy.parameters(agent_id=actor_id), self.policy.parameters(agent_id=critic_id)), 
                self.max_grad_norm[actor_id]
            )
        else:
          # Only clip actor gradients
          nn.utils.clip_grad_norm_(
              self.policy.parameters(agent_id=actor_id), 
              self.max_grad_norm[actor_id]
          )
        
        self.optimizers[actor_id].step()
      else:
        # Different optimizers for actor and critic
        # Always update actor
        self.optimizers[actor_id].zero_grad()
        loss.backward()  # retain_graph only if we need to update critic too
        nn.utils.clip_grad_norm_(self.policy.parameters(agent_id=actor_id), self.max_grad_norm[actor_id])
        self.optimizers[actor_id].step()
        if update_critic:
          self.optimizers[critic_id].zero_grad()
          nn.utils.clip_grad_norm_(self.policy.parameters(agent_id=critic_id), self.max_grad_norm[critic_id])
          self.optimizers[critic_id].step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
    
    num_updates = self.num_learning_epochs[actor_id] * self.num_mini_batches[actor_id]
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates

    self.storage[actor_id].clear()

    return surrogate_loss.item(), value_loss.item(), mean_entropy