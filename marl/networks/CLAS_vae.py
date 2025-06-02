import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from torch.distributions import Normal, TransformedDistribution, TanhTransform

"""
INPUT: full observation (joint angles, EE pose, object pose, obj_handle pose) (concatenated from all agents), 
        actions (as joint velocities) (concatenated from all agents)
OUTPUT: latent action space parameters (mu, logvar) (non-semantic meaning here is the wrench on the object)
"""
class CLASEncoder(nn.Module):
    """Encoder network that maps observations and actions to latent space"""
    
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Input: concatenated observations and actions from all agents
        input_dim = obs_dim + action_dim
        
        # 3 hidden layers as mentioned in paper
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean and log variance for latent distribution
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.logvar_layer.bias, -5.0)

        
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Full observation (concatenated from all agents)
            actions: Actions from all agents
        Returns:
            mu, logvar: Parameters of latent distribution (before tanh)
        """
        x = torch.cat([obs, actions], dim=-1)
        hidden = self.layers(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        # Clamp logvar to avoid extreme values
        # This is optional but recommended to prevent numerical instability during training
        # try turning this off if anything goes wrong during training
        logvar = logvar.clamp(min=-10, max=2)

        return mu, logvar

"""
INPUT: full observation o (joint angles, EE pose, object pose, obj_handle pose) (concatenated from all agents), 
        latent actions v_c (as joint velocities) (concatenated from all agents)
OUTPUT: original action
"""
class CLASDecoder(nn.Module):
    """Decoder network that maps latent actions to original action space"""
    
    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Input: observation + latent action
        input_dim = obs_dim + latent_dim
        
        # 3 hidden layers as mentioned in paper
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers
        self.mu_layer = nn.Linear(hidden_dim, action_dim)

        self.logvar_layer = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.logvar_layer.bias, -5.0)

        
    def forward(self, obs: torch.Tensor, latent_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Observation for specific agent
            latent_action: Latent action from shared policy (already tanh-squashed)
        Returns:
            mu, logvar: Parameters of action distribution (before tanh)
        """
        x = torch.cat([obs, latent_action], dim=-1)
        hidden = self.layers(x)
        mu = self.mu_layer(hidden)
        
        logvar = self.logvar_layer(hidden)
        # Clamp logvar to avoid extreme values
        # This is optional but recommended to prevent numerical instability during training
        # try turning this off if anything goes wrong during training
        logvar = logvar.clamp(min=-10, max=2)
    
        return mu, logvar

class CLASPrior(nn.Module):
    """Prior network p(v|o) - policy-like form"""
    
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 2 hidden layers for policy networks as mentioned in paper
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.logvar_layer.bias, -5.0)

        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu, logvar: Parameters of prior distribution (before tanh)
        """
        hidden = self.layers(obs)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        # Clamp logvar to avoid extreme values
        # This is optional but recommended to prevent numerical instability during training
        # try turning this off if anything goes wrong during training
        logvar = logvar.clamp(min=-10, max=2)
        
        return mu, logvar
    

class CLASVAE:
    """Complete CLAS VAE system for two-arm manipulation"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dictionary containing network configurations
        """
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Extract dimensions from config
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Calculate observation dimensions from your data structure
        self.robot0_obs_dim = 39
        self.robot1_obs_dim = 39
        self.shared_obs_dim = 17
        self.full_obs_dim = self.robot0_obs_dim + self.robot1_obs_dim + self.shared_obs_dim
        
        # Action dimensions (assuming 7 DOF for each robot arm)
        self.robot0_action_dim = 6
        self.robot1_action_dim = 6
        self.total_action_dim = self.robot0_action_dim + self.robot1_action_dim
        
        # Initialize networks
        self.encoder = CLASEncoder(
            obs_dim=self.full_obs_dim,
            action_dim=self.total_action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.decoder0 = CLASDecoder(
            obs_dim=self.full_obs_dim,
            latent_dim=self.latent_dim,
            action_dim=self.robot0_action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        
        self.decoder1 = CLASDecoder(
            obs_dim=self.full_obs_dim,
            latent_dim=self.latent_dim,
            action_dim=self.robot1_action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        
        self.prior = CLASPrior(
            obs_dim=self.full_obs_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        
        # Optimizer
        # Update all networks together (as they're coupled)
        all_params = (list(self.encoder.parameters()) + 
                     list(self.decoder0.parameters()) + 
                     list(self.decoder1.parameters()) + 
                     list(self.prior.parameters()))
        
        self.vae_optimizer = optim.Adam(all_params, lr=1e-4)
        
    
    def parse_observation(self, obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse observation dictionary into robot-specific and shared observations
        
        Returns:
            robot0_obs, robot1_obs, shared_obs, full_obs
        """
        # Robot 0 observations
        robot0_obs = torch.cat([
            torch.as_tensor(obs_dict['robot0_joint_pos'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot0_joint_pos_cos'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot0_joint_pos_sin'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot0_joint_vel'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot0_eef_pos'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot0_eef_quat'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot0_eef_quat_site'], dtype=torch.float32, device=self.device),
            # torch.as_tensor(obs_dict['robot0_gripper_qvel'], dtype=torch.float32, device=self.device),
        ], dim=-1)

        """
        obs_dict keys: dict_keys(
            ['angle', 'd', 'hole_pos', 'hole_quat', 
            'object-state', 'peg_quat', 'peg_to_hole', 'robot0_eef_pos', 
            
            'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_joint_pos', 
            'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 
            'robot0_proprio-state', 'robot1_eef_pos', 'robot1_eef_quat', 
            'robot1_eef_quat_site', 'robot1_joint_pos', 'robot1_joint_pos_cos', 
            'robot1_joint_pos_sin', 'robot1_joint_vel', 'robot1_proprio-state', 't'])
        """


        # Robot 1 observations
        robot1_obs = torch.cat([
            torch.as_tensor(obs_dict['robot1_joint_pos'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot1_joint_pos_cos'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot1_joint_pos_sin'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot1_joint_vel'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot1_eef_pos'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot1_eef_quat'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['robot1_eef_quat_site'], dtype=torch.float32, device=self.device),
            # torch.as_tensor(obs_dict['robot1_gripper_qvel'], dtype=torch.float32, device=self.device),
        ], dim=-1)
        
        # Shared observations (task-related)
        shared_obs = torch.cat([
            torch.as_tensor(obs_dict['angle'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['d'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['t'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['hole_pos'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['hole_quat'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['peg_quat'], dtype=torch.float32, device=self.device),
            torch.as_tensor(obs_dict['peg_to_hole'], dtype=torch.float32, device=self.device),
        ], dim=-1)
        
        # Full observation (concatenated)
        full_obs = torch.cat([robot0_obs, robot1_obs, shared_obs], dim=-1)
        
        return robot0_obs, robot1_obs, shared_obs, full_obs
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def tanh_normal(self, mu, logvar):
        base = Normal(mu, torch.exp(0.5 * logvar))
        # cache_size=1 speeds up the log‐jacobian calculation
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])
    
    def vae_loss(self, obs_dict: Dict, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        
        if torch.isnan(mu_enc).any() or torch.isnan(logvar_enc).any():
            raise RuntimeError("NaN detected in encoder output")

        latent_pre_tanh = self.reparameterize(mu_enc, logvar_enc)
        latent_action = torch.tanh(latent_pre_tanh)

        mu_dec_0, logvar_dec_0 = self.decoder0(full_obs, latent_action)
        mu_dec_1, logvar_dec_1 = self.decoder1(full_obs, latent_action)

        if torch.isnan(mu_dec_0).any() or torch.isnan(logvar_dec_0).any():
            raise RuntimeError("NaN in decoder0 output")
        if torch.isnan(mu_dec_1).any() or torch.isnan(logvar_dec_1).any():
            raise RuntimeError("NaN in decoder1 output")

        dist0 = self.tanh_normal(mu_dec_0, logvar_dec_0)
        dist1 = self.tanh_normal(mu_dec_1, logvar_dec_1)

        # Split actions
        robot0_action = actions[:, :self.robot0_action_dim]
        robot1_action = actions[:, self.robot0_action_dim:]

        # *** CLAMP the buffered actions into (-1+eps, 1-eps) ***
        eps = 1e-4
        robot0_action = robot0_action.clamp(min=-1.0 + eps, max=1.0 - eps)
        robot1_action = robot1_action.clamp(min=-1.0 + eps, max=1.0 - eps)

        logp0 = dist0.log_prob(robot0_action).sum(dim=-1)
        logp1 = dist1.log_prob(robot1_action).sum(dim=-1)

        if torch.isnan(logp0).any() or torch.isnan(logp1).any():
            raise RuntimeError("NaN in log-probability")

        recon_loss = -(logp0 + logp1).mean()

        mu_prior, logvar_prior = self.prior(full_obs)
        if torch.isnan(mu_prior).any() or torch.isnan(logvar_prior).any():
            raise RuntimeError("NaN in prior output")

        # KL: q(v|o,u) vs p(v|o)
        kl_loss = -0.5 * torch.sum(
            1 + logvar_enc - logvar_prior -
            ((mu_enc - mu_prior).pow(2) + logvar_enc.exp()) / logvar_prior.exp(),
            dim=-1
        ).mean()

        if torch.isnan(kl_loss):
            raise RuntimeError("NaN in kl_loss")

        total_loss = recon_loss + kl_loss
        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}
    
    def train_step(self, obs_dict: Dict, actions: torch.Tensor) -> Dict[str, float]:
        """Single training step for the VAE"""
        
        # Compute losses
        losses = self.vae_loss(obs_dict, actions)
        
        self.vae_optimizer.zero_grad()
        losses['total_loss'].backward()
        self.vae_optimizer.step()
        
        # Convert to float for logging
        return {k: v.item() for k, v in losses.items()}
    
    
    def decode_actions(self, obs_dict: Dict, latent_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent action to robot-specific actions"""
        
        _, _, _, full_obs = self.parse_observation(obs_dict)
        
        full_obs = full_obs.to(self.device)
        
        latent_action = latent_action.to(self.device)
        
        with torch.no_grad():
            mu_0, _ = self.decoder0(full_obs, latent_action)
            mu_1, _ = self.decoder1(full_obs, latent_action)
            
            # Apply tanh to get final actions
            robot0_action = torch.tanh(mu_0)
            robot1_action = torch.tanh(mu_1)
        
        return robot0_action, robot1_action
    
    
# unit tests for CLASVAE as sanity check
if __name__ == "__main__":
    # Example usage
    config = {
        'latent_dim': 16,
        'hidden_dim': 256
    }
    
    vae = CLASVAE(config)
    
    # Test different batch sizes
    batch_sizes = [1, 8, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create batched dummy observations
        obs_dict = {
            'robot0_joint_pos': np.random.rand(batch_size, 7),
            'robot0_joint_pos_cos': np.random.rand(batch_size, 7),
            'robot0_joint_pos_sin': np.random.rand(batch_size, 7),
            'robot0_joint_vel': np.random.rand(batch_size, 7),
            'robot0_eef_pos': np.random.rand(batch_size, 3),
            'robot0_eef_quat': np.random.rand(batch_size, 4),
            'robot0_eef_quat_site': np.random.rand(batch_size, 4),
            'robot0_gripper_qpos': np.random.rand(batch_size, 2),
            'robot0_gripper_qvel': np.random.rand(batch_size, 2),
            
            'robot1_joint_pos': np.random.rand(batch_size, 7),
            'robot1_joint_pos_cos': np.random.rand(batch_size, 7),
            'robot1_joint_pos_sin': np.random.rand(batch_size, 7),
            'robot1_joint_vel': np.random.rand(batch_size, 7),
            'robot1_eef_pos': np.random.rand(batch_size, 3),
            'robot1_eef_quat': np.random.rand(batch_size, 4),
            'robot1_eef_quat_site': np.random.rand(batch_size, 4),
            'robot1_gripper_qpos': np.random.rand(batch_size, 2),
            'robot1_gripper_qvel': np.random.rand(batch_size, 2),
            
            'pot_pos': np.random.rand(batch_size, 3),
            'pot_quat': np.random.rand(batch_size, 4),
            'handle0_xpos': np.random.rand(batch_size, 3),
            'handle1_xpos': np.random.rand(batch_size, 3),
            'gripper0_to_handle0': np.random.rand(batch_size, 3),
            'gripper1_to_handle1': np.random.rand(batch_size, 3)
        }
        
        # Create batched actions
        actions = torch.tanh(torch.randn((batch_size, 14)))  # 14 total action dimensions
        
        try:
            # Test training step
            losses = vae.train_step(obs_dict, actions)
            print(f"  Training step successful - Losses: {losses}")
            
            
            
            # Test action decoding
            latent_action = torch.randn((batch_size, config['latent_dim']))
            robot0_action, robot1_action = vae.decode_actions(obs_dict, latent_action)
            
            # Verify shapes
            expected_shape = (batch_size, 7)  # Assuming 7 DOF per robot
            assert robot0_action.shape == expected_shape, f"Robot0 shape mismatch: {robot0_action.shape} != {expected_shape}"
            assert robot1_action.shape == expected_shape, f"Robot1 shape mismatch: {robot1_action.shape} != {expected_shape}"
            
            print(f"  Action decoding successful - Shapes: robot0={robot0_action.shape}, robot1={robot1_action.shape}")
            
            # Test observation parsing
            robot0_obs, robot1_obs, shared_obs, full_obs = vae.parse_observation(obs_dict)
            print(f"  Observation parsing successful - Full obs shape: {full_obs.shape}")
            
            # Test VAE loss computation (without optimizer step)
            loss_dict = vae.vae_loss(obs_dict, actions)
            print(f"  VAE loss computation successful - Losses: {[f'{k}: {v.item():.4f}' for k, v in loss_dict.items()]}")
            
        except Exception as e:
            print(f"  ERROR with batch size {batch_size}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("ADDITIONAL TESTS")
    print("="*50)
    
    # Test with different action dimensions
    print("\nTesting action dimension consistency:")
    batch_size = 16
    obs_dict = {
        'robot0_joint_pos': np.random.rand(batch_size, 7),
        'robot0_joint_pos_cos': np.random.rand(batch_size, 7),
        'robot0_joint_pos_sin': np.random.rand(batch_size, 7),
        'robot0_joint_vel': np.random.rand(batch_size, 7),
        'robot0_eef_pos': np.random.rand(batch_size, 3),
        'robot0_eef_quat': np.random.rand(batch_size, 4),
        'robot0_eef_quat_site': np.random.rand(batch_size, 4),
        'robot0_gripper_qpos': np.random.rand(batch_size, 2),
        'robot0_gripper_qvel': np.random.rand(batch_size, 2),
        
        'robot1_joint_pos': np.random.rand(batch_size, 7),
        'robot1_joint_pos_cos': np.random.rand(batch_size, 7),
        'robot1_joint_pos_sin': np.random.rand(batch_size, 7),
        'robot1_joint_vel': np.random.rand(batch_size, 7),
        'robot1_eef_pos': np.random.rand(batch_size, 3),
        'robot1_eef_quat': np.random.rand(batch_size, 4),
        'robot1_eef_quat_site': np.random.rand(batch_size, 4),
        'robot1_gripper_qpos': np.random.rand(batch_size, 2),
        'robot1_gripper_qvel': np.random.rand(batch_size, 2),
        
        'pot_pos': np.random.rand(batch_size, 3),
        'pot_quat': np.random.rand(batch_size, 4),
        'handle0_xpos': np.random.rand(batch_size, 3),
        'handle1_xpos': np.random.rand(batch_size, 3),
        'gripper0_to_handle0': np.random.rand(batch_size, 3),
        'gripper1_to_handle1': np.random.rand(batch_size, 3)
    }
    
    # Test encoding and decoding consistency
    print("\nTesting encoding-decoding consistency:")
    original_actions = torch.tanh(torch.randn((batch_size, 14)))
    
    # Encode actions to latent space
    robot0_obs, robot1_obs, shared_obs, full_obs = vae.parse_observation(obs_dict)
    full_obs = full_obs.to(vae.device)
    original_actions = original_actions.to(vae.device)
    
    with torch.no_grad():
        # Encode
        mu_enc, logvar_enc = vae.encoder(full_obs, original_actions)
        latent_action = torch.tanh(vae.reparameterize(mu_enc, logvar_enc))
        
        # Decode
        decoded_robot0, decoded_robot1 = vae.decode_actions(obs_dict, latent_action)
        decoded_actions = torch.cat([decoded_robot0, decoded_robot1], dim=-1)
        
        # Compute reconstruction error
        recon_error = torch.mean(torch.abs(original_actions - decoded_actions))
        print(f"  Reconstruction error: {recon_error.item():.6f}")
        
        if recon_error.item() < 2.0:  # Reasonable threshold for random initialization
            print("  ✓ Reconstruction error within acceptable range")
        else:
            print("  ⚠ High reconstruction error (expected for untrained model)")
    
    print("\nAll batch operation tests completed!")