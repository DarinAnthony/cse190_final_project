import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Any, Dict, List, Optional, Tuple
import copy
from collections import deque
import random

class ReplayBuffer:
    """Replay buffer for SAC"""
    def __init__(self, capacity, obs_dim, latent_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.empty((capacity, obs_dim),     device=device)
        self.lat = torch.empty((capacity, latent_dim),  device=device)
        self.rew = torch.empty((capacity, 1),           device=device)
        self.nxt = torch.empty((capacity, obs_dim),     device=device)
        self.done = torch.empty((capacity, 1),          device=device)
        self.ptr = 0 
        self.full = False
        
    def push(self, obs, latent, reward, next_obs, done):
        idx = self.ptr
        self.obs[idx]  = obs
        self.lat[idx]  = latent
        self.rew[idx]  = reward
        self.nxt[idx]  = next_obs
        self.done[idx] = done
        self.ptr = (idx + 1) % self.capacity
        if idx == self.capacity - 1:
            self.full = True
        
    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.ptr
        idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return (self.obs[idx],
                self.lat[idx],
                self.rew[idx],
                self.nxt[idx],
                self.done[idx])
    
    def __len__(self):
        return self.capacity if self.full else self.ptr

class LatentPolicy(nn.Module):
    """Policy network that outputs actions in the latent space"""
    def __init__(self, obs_dim, latent_dim, hidden_dim=256, device=None):
        super().__init__()
        
        self.device = device
        
        # Policy network (2 hidden layers as mentioned in paper)
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_std_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Action bounds
        self.action_scale = torch.tensor(1.0, device=self.device)
        self.action_bias = torch.tensor(0.0, device=self.device)
        
    def forward(self, obs):
        hidden = self.layers(obs)
        mu = self.mu_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std
    
    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Pre-tanh distribution
        dist = Normal(mu, std)
        x_t = dist.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(x_t)
        # Enforcing action bounds through tanh squashing
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, torch.tanh(mu)

class QNetwork(nn.Module):
    """Q-network for SAC"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.layers(x)

class CLASSAC:
    """SAC agent that operates in the latent action space"""
    def __init__(self, vae_model, obs_dim, latent_dim, config):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Store the pre-trained VAE
        self.vae = vae_model
            
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)
        self.lr = config.get('lr', 3e-4)
        
        # Initialize networks
        self.policy = LatentPolicy(obs_dim, latent_dim, device=self.device).to(self.device)
        self.q1 = QNetwork(obs_dim, latent_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, latent_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        
        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.target_entropy = -latent_dim  # -dim(A)
        
    def _batch_one_observation(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Take a single observation dictionary (values are 1D or 2D arrays)
        and turn every value into a batch of size 1, i.e. shape (1, *).

        Args:
            obs_dict: single observation, e.g. {'camera': array(D,), 'proprio': array(D2,), …}

        Returns:
            batched_obs: {'camera': array(1, D), 'proprio': array(1, D2), …}
        """
        batched = {}
        for key, arr in obs_dict.items():
            a = np.asarray(arr)
            if a.ndim == 1:
                batched[key] = a[np.newaxis, ...]       # shape (1, D)
            else:
                # If already has a batch‐like dim but with size >1, we assume it's okay.
                # e.g. shape (1, D, H, W) stays as is.  Otherwise, you can also do:
                batched[key] = a if a.shape[0] != 1 else a
        return batched
        
    def select_action(self, obs_dict, evaluate=False):
        # 1) batch the single observation:
        batched = self._batch_one_observation(obs_dict)  
        #    now every batched[key] has shape (1, …)

        # 2) parse through VAE to get a (1, obs_dim) tensor
        _, _, _, full_obs = self.vae.parse_observation(batched)
        full_obs = full_obs.to(self.device)  

        # 3) run the policy network
        with torch.no_grad():
            if evaluate:
                mu, _ = self.policy(full_obs)         # mu is (1, latent_dim)
                latent_action = torch.tanh(mu)         # still (1, latent_dim)
            else:
                latent_action, _, _ = self.policy.sample(full_obs)

            # 4) decode back to per-robot actions (each should be (1, 6))
            r0, r1 = self.vae.decode_actions(batched, latent_action)
            
            # 5) concatenate into a (1, 12) tensor
            actions = torch.cat([r0, r1], dim=-1)   # shape: (1, 12)

        # 6) detach & convert to numpy, ensuring proper shape
        action_np = actions.squeeze(0).detach().cpu().numpy()  # shape: (12,)
        latent_action_np = latent_action.squeeze(0).detach().cpu().numpy()  # shape: (latent_dim,)
        
        # Ensure action_np is a proper 1D array with correct length
        if action_np.ndim == 0:  # scalar case
            raise ValueError(f"Action became scalar: {action_np}")
        
        # The environment expects shape (1, 12) for vectorized env
        action_for_env = action_np.reshape(1, -1)
        
        return action_for_env, latent_action_np

    
    def update_q_networks(self, obs_batch, latent_action_batch, reward_batch, 
                         next_obs_batch, done_batch):
        """Update Q-networks"""
        with torch.no_grad():
            # Sample next actions
            next_latent_action, next_log_prob, _ = self.policy.sample(next_obs_batch)
            
            # Compute target Q values
            q1_next_target = self.q1_target(next_obs_batch, next_latent_action)
            q2_next_target = self.q2_target(next_obs_batch, next_latent_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        
        # Current Q values
        q1_value = self.q1(obs_batch, latent_action_batch)
        q2_value = self.q2(obs_batch, latent_action_batch)
        
        # Q losses
        q1_loss = F.mse_loss(q1_value, next_q_value)
        q2_loss = F.mse_loss(q2_value, next_q_value)
        
        # Update Q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.)
        self.q2_optimizer.step()
        
        return q1_loss.item(), q2_loss.item()
    
    def update_policy(self, obs_batch):
        """Update policy network"""
        # Sample actions
        latent_action, log_prob, _ = self.policy.sample(obs_batch)
        
        # Q values
        q1_value = self.q1(obs_batch, latent_action)
        q2_value = self.q2(obs_batch, latent_action)
        min_q_value = torch.min(q1_value, q2_value)
        
        # Policy loss
        policy_loss = ((self.alpha * log_prob) - min_q_value).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
        self.policy_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        
        return policy_loss.item(), alpha_loss.item()
    
    def update_target_networks(self):
        """Soft update of target networks"""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(self, replay_buffer, batch_size=256):
        """Single training step"""
        if len(replay_buffer) < batch_size:
            return None
        
        # Sample from replay buffer
        obs, lat, rew, nxt, done = replay_buffer.sample(batch_size)
        
        q1_loss, q2_loss = self.update_q_networks(obs, lat, rew, nxt, done)
        policy_loss, alpha_loss = self.update_policy(obs)
        self.update_target_networks()
        
        return {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'policy_loss': policy_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item()
        }
        
    # -----------------------------------------
    # 1.  Create a "checkpoint" dict and save
    # -----------------------------------------
    def save_checkpoint(self, ckpt_path: str, episode: int, avg_reward: float):
        """
        Persist policy, critics, alpha, optimizers, and metadata.
        """

        checkpoint = {
            # Networks
            "policy_state": self.policy.state_dict(),
            "q1_state":     self.q1.state_dict(),
            "q2_state":     self.q2.state_dict(),
            # Soft-Q target nets
            "q1_target":    self.q1_target.state_dict(),
            "q2_target":    self.q2_target.state_dict(),
            # Entropy temperature
            "log_alpha":    self.log_alpha.detach().cpu(),
            # Optimizers (optional but handy for resume-training)
            "policy_opt":   self.policy_optimizer.state_dict(),
            "q1_opt":       self.q1_optimizer.state_dict(),
            "q2_opt":       self.q2_optimizer.state_dict(),
            "alpha_opt":    self.alpha_optimizer.state_dict(),
            # Book-keeping
            "episode":      episode,
            "avg_reward":   avg_reward,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"✔ Saved checkpoint to {ckpt_path} (Ep {episode}, R̄={avg_reward:.2f})")
        
        
    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.policy.load_state_dict(ckpt["policy_state"])
        self.q1.load_state_dict(ckpt["q1_state"])
        self.q2.load_state_dict(ckpt["q2_state"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])

        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

        # If you want to resume training:
        self.policy_optimizer.load_state_dict(ckpt["policy_opt"])
        self.q1_optimizer.load_state_dict(ckpt["q1_opt"])
        self.q2_optimizer.load_state_dict(ckpt["q2_opt"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_opt"])

        print(f"✔ Loaded checkpoint from {ckpt_path} (Ep {ckpt['episode']}, R̄={ckpt['avg_reward']:.2f})")
        return ckpt["episode"], ckpt["avg_reward"]