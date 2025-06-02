import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from marl.networks.CLAS_vae import CLASVAE

class CLASVAEWithBeta(CLASVAE):
    def __init__(self, config, beta_schedule='linear'):
        super().__init__(config)
        self.beta = 0.0
        self.beta_schedule = beta_schedule
        self.max_beta = config.get('max_beta', 1.0)
        self.beta_warmup_steps = config.get('beta_warmup_steps', 50000)
        
    def update_beta(self, step):
        if self.beta_schedule == 'linear':
            self.beta = min(self.max_beta, step / self.beta_warmup_steps)
        elif self.beta_schedule == 'constant':
            self.beta = self.max_beta
            
    def vae_loss(self, obs_dict, actions):
        """
        Compute VAE loss as described in the paper (Equation 2)
        
        Args:
            obs_dict: Observation dictionary
            actions: Actions taken by both robots [robot0_action, robot1_action] (already tanh-squashed)
        """
        robot0_obs, robot1_obs, shared_obs, full_obs = self.parse_observation(obs_dict)
        
        # Move to device
        robot0_obs = robot0_obs.to(self.device)
        robot1_obs = robot1_obs.to(self.device)
        shared_obs = shared_obs.to(self.device)
        full_obs = full_obs.to(self.device)
        actions = actions.to(self.device)
        
        # Encoder: q(v|o,u) - get latent distribution parameters
        mu_enc, logvar_enc = self.encoder(full_obs, actions)
        
        # Sample latent action (before tanh)
        latent_pre_tanh = self.reparameterize(mu_enc, logvar_enc)
        latent_action = torch.tanh(latent_pre_tanh)
        
        # Decoders: p(u|o,v) - get action distribution parameters
        mu_dec_0, logvar_dec_0 = self.decoder0(
            full_obs, 
            latent_action
        )
        mu_dec_1, logvar_dec_1 = self.decoder1(
            full_obs, 
            latent_action
        )
        
        dist0 = self.tanh_normal(mu_dec_0, logvar_dec_0)
        dist1 = self.tanh_normal(mu_dec_1, logvar_dec_1)
        
        # Prior: p(v|o)
        mu_prior, logvar_prior = self.prior(full_obs)
        
        # Split actions
        robot0_action = actions[:, :self.robot0_action_dim]
        robot1_action = actions[:, self.robot0_action_dim:]
        
        # *** CLAMP the buffered actions into (-1+eps, 1-eps) ***
        eps = 1e-4
        robot0_action = robot0_action.clamp(min=-1.0 + eps, max=1.0 - eps)
        robot1_action = robot1_action.clamp(min=-1.0 + eps, max=1.0 - eps)
        
        logp0 = dist0.log_prob(robot0_action).sum(dim=-1)   # sum over action dims
        logp1 = dist1.log_prob(robot1_action).sum(dim=-1)
        
        recon_loss = -(logp0 + logp1).mean()
        
        # KL divergence between encoder and prior (both before tanh)
        # KL(q(v|o,u) || p(v|o))
        kl_loss = -0.5 * torch.sum(
            1 + logvar_enc - logvar_prior - 
            ((mu_enc - mu_prior).pow(2) + logvar_enc.exp()) / logvar_prior.exp(),
            dim=-1
        ).mean()
        
        total_loss = recon_loss + self.beta * kl_loss
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }