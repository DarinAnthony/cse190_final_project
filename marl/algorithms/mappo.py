from typing import Dict, Any

from marl.algorithms.ppo import PPO
from marl.storage.rollout_storage import RolloutStorage

from torch import optim

class MAPPO(PPO):
    """ Multi-Agent Proximal Policy Optimization 
    
    This class assumes that there are two agents and a single shared critic.
    
    """
    def __init__(
        self,
        policy,
        agent_hyperparams: Dict[str, Dict[str, Any]],
        normalize_advantage_per_mini_batch: bool = False,
        actor_critic_mapping: Dict[str, str] = None,
    ):
        super().__init__(policy, agent_hyperparams, normalize_advantage_per_mini_batch, mappo=True)

        self.actor_critic_mapping = actor_critic_mapping
    
    def _setup_optimizers(self):
        """ Setup optimizers for each agent and shared critic """
        # Set up optimizers for all actor networks
        for actor_id in self.actors:
            print(f"SETTING UP OPTIMIZER FOR ACTOR NETWORK {actor_id}")
            self.optimizers[actor_id] = optim.Adam(
                self.policy.parameters(agent_id=actor_id),
                lr=self.learning_rate[actor_id]
            )
        
        # Set up optimizer for shared critic
        critic_id = self.critics[0]  # Assuming single shared critic
        print(f"SETTING UP OPTIMIZER FOR SHARED CRITIC NETWORK {critic_id}")
        self.optimizers[critic_id] = optim.Adam(
            self.policy.parameters(agent_id=critic_id),
            lr=self.learning_rate[critic_id]
        )

    
    def _init_storage(self, num_envs: int, num_transitions_per_env: int):
        """
        Initialize storage for agents.
        
        Args:
            num_envs: Number of environments
            num_transitions_per_env: Number of transitions to rollout
        """
        for actor_id in self.actors:
            critic_id = self.actor_critic_mapping[actor_id]

            self.storage[actor_id] = RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            actor_obs_dim=self.policy.components[actor_id].network_kwargs["actor_obs_dim"],
            critic_obs_dim=self.policy.components[critic_id].network_kwargs["critic_obs_dim"],
            action_dim=self.policy.components[actor_id].network_kwargs["num_actions"],
            device=self.policy.device
        )
    
    def act(self, actor_obs, critic_obs=None):
        """
        Execute actions for all agents.
        
        Args:
            actor_obs: Observations for actor (can be dict mapping agent_id->obs or single obs)
            critic_obs: Observations for critic (can be dict mapping agent_id->obs or single obs)
        
        Returns:
            actions: Actions for all agents
        """
        all_actions = {}
        for actor_id in self.actors:
            critic_id = self.actor_critic_mapping[actor_id]

            self.transitions[actor_id].actions = self.policy.act(actor_obs[actor_id], agent_id=actor_id).detach()
            self.transitions[actor_id].values = self.policy.evaluate(critic_obs[critic_id], agent_id=critic_id).detach()
            self.transitions[actor_id].actions_log_prob = self.policy.get_actions_log_prob(self.transitions[actor_id].actions, actor_id).detach()
            self.transitions[actor_id].action_mean = self.policy.get_action_mean(actor_id).detach() 
            self.transitions[actor_id].action_sigma = self.policy.get_action_std(actor_id).detach()
            self.transitions[actor_id].actor_observations = actor_obs[actor_id]
            self.transitions[actor_id].critic_observations = critic_obs[critic_id]
            all_actions[actor_id] = self.transitions[actor_id].actions

        return all_actions

    def compute_returns(self, last_critic_obs) -> None:
      """
      Compute returns for agents.
      
      Args:
          last_critic_obs: Last critic observations 
      """
      for actor_id in self.actors:
        critic_id = self.actor_critic_mapping[actor_id]
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
        update_critic = False
        for actor_id in self.actors:
            critic_id = self.actor_critic_mapping[actor_id]
            surrogate_loss, value_loss, entropy_loss = self._update_single_agent(actor_id, critic_id, update_critic)
            all_loss_dicts[actor_id] = {
                'surrogate_loss': surrogate_loss,

            }
            update_critic = True
        all_loss_dicts = {
            "entropy_loss": entropy_loss,
            "value_loss": value_loss
        }
        return all_loss_dicts
        
    
