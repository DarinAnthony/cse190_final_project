import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import mujoco
from mujoco import _functions as mjf

from marl.agents.base_marl import BaseMARLAgent
from marl.policies.base_policy import BasePolicy
from marl.algorithms.base import BaseAlgorithm
from marl.storage.VAE_rBuffer import VAEBuf
from marl.networks.CLAS_vae import CLASVAE  # Assuming CLASVAE is defined in this module

class CLASVAEAgent(BaseMARLAgent):
    """
    CLAS VAE Agent that extends BaseMARLAgent for VAE-based multi-agent learning.
    
    This agent:
    1. Collects experience in a VAE-specific replay buffer
    2. Trains a CLAS VAE model to learn latent action representations
    3. Uses the trained VAE for coordinated multi-agent control
    """

    def __init__(
        self,
        env: Any,
        policy: BasePolicy,
        observation_config: Dict[str, List[str]],
        num_transitions_per_env: int,
        normalize_observations: bool,
        vae_config: Dict[str, Any],
        vae_buffer_size: int = 100000,
        vae_batch_size: int = 128,
        vae_train_freq: int = 1,
        min_buffer_size: int = 1000,
        algorithm: Optional[BaseAlgorithm] = None,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            vae_config: Configuration dictionary for CLASVAE
            vae_buffer_size: Size of VAE replay buffer
            vae_batch_size: Batch size for VAE training
            vae_train_freq: Frequency of VAE training (every N environment steps)
            min_buffer_size: Minimum buffer size before starting VAE training
        """
        # Initialize base class
        super().__init__(
            env=env,
            policy=policy,
            algorithm=algorithm,
            observation_config=observation_config,
            num_transitions_per_env=num_transitions_per_env,
            normalize_observations=normalize_observations,
            logger=logger,
            device=device
        )
        
        # VAE-specific initialization
        self.vae_config = vae_config
        self.vae_batch_size = vae_batch_size
        self.vae_train_freq = vae_train_freq
        self.min_buffer_size = min_buffer_size
        
        # Initialize VAE buffer
        self.vae_buffer = VAEBuf(capacity=vae_buffer_size)
        self.vae_eval_buffer = VAEBuf(capacity= vae_buffer_size)
        
        # Initialize CLAS VAE
        self.vae = CLASVAE(config=vae_config)
        
        # Training counters
        self.step_count = 0
        self.vae_training_steps = 0
        
        # Training metrics
        self.vae_losses = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }

    def store_transition(self, buffer: VAEBuf, obs_dict: Dict[str, np.ndarray], actions: np.ndarray):
        """
        Store a transition in the VAE buffer.
        
        Args:
            obs_dict: Raw observation dictionary from environment
            actions: Combined actions from both robots [robot0_action, robot1_action]
        """
        # Ensure actions are in the right format (numpy array)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        # Store in VAE buffer
        buffer.push(obs_dict, actions)
        
    def _prefill_buffer(self,
                        buffer: VAEBuf,
                        prefill_size: int = 50000,
                        max_episode_steps: int = 200) -> None:
        """
        Step the env with a hand-coded / random policy until
        self.vae_buffer contains `prefill_size` transitions.
        """
        if self.logger:
            self.logger.info(f"Prefilling VAE buffer with {prefill_size} transitions")

        
        obs_dict, infos = self.env.reset()  # Reset and weld the robots to the handle
        
        steps_in_ep = 0

        pbar = tqdm(total=prefill_size, desc="Prefill", unit="step")
        
        while len(self.vae_buffer) < prefill_size:
            # ----- cheap exploration policy -----
            # completely random joint velocities in [-1, 1]
            upward_u = np.random.uniform(-1.0, 1.0, size=(1, 12))
            # rand_u = np.zeros((1, 14))
            # Coordinated upward movement
            # upward_u = np.zeros((1, 14))
            # upward_u[0, 2] = 0.5  # Robot 0 vertical joint
            # upward_u[0, 9] = 0.5  # Robot 1 vertical joint (assuming symmetric)
            # OR: a scripted grab-and-weld controller you already have
            # rand_u = my_scripted_weld_controller(obs_dict)

            next_obs, reward, done, truncated, info = self.env.step(upward_u)
            self.store_transition(buffer, obs_dict, upward_u)
            pbar.update(1)

            obs_dict = next_obs
            steps_in_ep += 1

            if done or steps_in_ep >= max_episode_steps:
                obs_dict, infos = self.env.reset()
                steps_in_ep = 0

        pbar.close()
        if self.logger:
            self.logger.info("Prefill done ✓")
            
            
    def _prefill_buffer_diverse(self, buffer: VAEBuf, prefill_size: int = 50000, max_episode_steps: int = 200) -> None:
        """Collect data with various exploration strategies"""
        strategies = [
            lambda: np.random.uniform(-1, 1, 12),  # Random
            lambda: np.random.normal(0, 0.3, 12),  # Gaussian noise
            lambda: np.concatenate([np.random.uniform(-1, 1, 6), np.zeros(6)]),  # One arm
            lambda: np.concatenate([np.zeros(6), np.random.uniform(-1, 1, 6)]),  # Other arm
        ]
        
        
        pbar = tqdm(total=prefill_size, desc="Prefill", unit="step")
        
        while len(buffer) < prefill_size:
            obs_dict, _ = self.env.reset()
            strategy = np.random.choice(strategies)
            
            for step in range(max_episode_steps):
                action = strategy()
                action = np.clip(action, -1, 1).reshape(1, -1)
                next_obs, _, done, _, _ = self.env.step(action)
                pbar.update(1)
                self.store_transition(buffer, obs_dict, action)
                obs_dict = next_obs
                if done or step > max_episode_steps:
                    break
            
        pbar.close()
        if self.logger:
            self.logger.info("Prefill done ✓")

    def train_vae_step(self) -> Dict[str, float]:
        """
        Perform one VAE training step if buffer has enough samples.
        
        Returns:
            Dictionary of training losses, or empty dict if training was skipped
        """
        if not self.vae_buffer.is_ready(self.min_buffer_size):
            if self.logger:
                self.logger.info("VAE buffer not ready for training, skipping step")
            return {}
        
        obs_dicts, actions = self.vae_buffer.sample_batch(self.vae_batch_size)
            
        # Convert list of obs_dicts to batched format
        batched_obs_dict = self._batch_observations(obs_dicts)
        
        # Train VAE
        losses = self.vae.train_step(batched_obs_dict, actions)
        
        # Update training counter
        self.vae_training_steps += 1
        
        # Store losses for logging
        for key, value in losses.items():
            self.vae_losses[key].append(value)
        
        try:
            # Sample batch from buffer
            obs_dicts, actions = self.vae_buffer.sample_batch(self.vae_batch_size)
            
            # Convert list of obs_dicts to batched format
            batched_obs_dict = self._batch_observations(obs_dicts)
            
            # Train VAE
            losses = self.vae.train_step(batched_obs_dict, actions)
            
            # Update training counter
            self.vae_training_steps += 1
            
            # Store losses for logging
            for key, value in losses.items():
                self.vae_losses[key].append(value)
        
            return losses
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"VAE training step failed: {str(e)}")
            return {}

    def _batch_observations(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Convert list of observation dictionaries to batched format.
        
        Args:
            obs_list: List of observation dictionaries
            
        Returns:
            Dictionary with batched observations
        """
        if not obs_list:
            return {}
        
        batched_obs = {}
        for key in obs_list[0].keys():
            # Stack observations across batch dimension
            arr = np.stack([obs[key] for obs in obs_list], axis=0)
            
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr[:, 0, :]    # now (B, D)

            batched_obs[key] = arr
        
        return batched_obs

    def get_latent_actions(self, obs_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Get latent actions from the VAE prior for coordination.
        
        Args:
            obs_dict: Current observation dictionary
            
        Returns:
            Sampled latent actions for coordination
        """
        # Parse observations
        _, _, _, full_obs = self.vae.parse_observation(obs_dict)
        
        with torch.no_grad():
            # Sample from prior p(v|o)
            mu_prior, logvar_prior = self.vae.prior(full_obs)
            latent_pre_tanh = self.vae.reparameterize(mu_prior, logvar_prior)
            latent_action = torch.tanh(latent_pre_tanh)
        
        return latent_action

    def decode_coordinated_actions(
        self, 
        obs_dict: Dict[str, np.ndarray], 
        latent_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent actions to robot-specific actions using trained VAE.
        
        Args:
            obs_dict: Current observations
            latent_action: Latent coordination signal
            
        Returns:
            Tuple of (robot0_actions, robot1_actiocns)
        """
        return self.vae.decode_actions(obs_dict, latent_action)

    def learn(self,
          prefill_size: int     = 10000,
          vae_updates:   int     = 100000,
          max_episode_steps: int = 500) -> None:
        """
        1) Collect `prefill_size` transitions with a random / scripted controller
        2) Run exactly `vae_updates` gradient steps on the VAE
        (sampling fresh batches from the filled buffer)
        """

        # ------------- Phase 1: prefilling ----------
        self._prefill_buffer_diverse(self.vae_buffer, prefill_size * 10, max_episode_steps)
        self._prefill_buffer_diverse(self.vae_eval_buffer, prefill_size, max_episode_steps)
        
        latent_dims_to_test = [4, 6, 8, 12, 16]
        hidden_dims_to_test = [128, 256, 512]  # Currently 256
        num_layers_to_test = [2, 3, 4]

        print("now training the VAE with {} updates".format(vae_updates))
        # ------------- Phase 2: VAE updates only ----------
        for update in range(vae_updates):
            losses = self.train_vae_step()           # draws a batch each call
            if self.logger and update % 500 == 0:
                self.logger.info(
                    f"VAE-update {update}/{vae_updates} "
                    f"| total: {losses.get('total_loss', 0):.4f} "
                    f"| recon: {losses.get('recon_loss', 0):.4f} "
                    f"| KL:    {losses.get('kl_loss', 0):.4f}"
                )
                
        self.eval_mode()
        eval_dict = self.evaluate_vae(self.vae, self.vae_eval_buffer, 50)
        if self.logger:
            self.logger.info(
                f"VAE evaluation: "
                f"recon_loss={eval_dict.get('val_recon_loss', 0):.4f}, "
                f"kl_loss={eval_dict.get('val_kl', 0):.4f}, "
                f"mse={eval_dict.get('val_mse', 0):.4f}"
            )

        # ------------- save weights for later RL ----------
        self.save_vae("clas_vae_prefilled.pt")



    ################################################
    ############### Utility Methods
    ################################################



    def save_vae(self, path: str):
        """Save the trained VAE model"""
        torch.save({
            'encoder': self.vae.encoder.state_dict(),
            'decoder0': self.vae.decoder0.state_dict(),
            'decoder1': self.vae.decoder1.state_dict(),
            'prior': self.vae.prior.state_dict(),
            'vae_config': self.vae_config,
            'training_steps': self.vae_training_steps
        }, path)
        
        if self.logger:
            self.logger.info(f"VAE model saved to {path}")

    def load_vae(self, path: str):
        """Load a pre-trained VAE model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.vae.encoder.load_state_dict(checkpoint['encoder'])
        self.vae.decoder0.load_state_dict(checkpoint['decoder0'])
        self.vae.decoder1.load_state_dict(checkpoint['decoder1'])
        self.vae.prior.load_state_dict(checkpoint['prior'])
        self.vae_training_steps = checkpoint.get('training_steps', 0)
        
        if self.logger:
            self.logger.info(f"VAE model loaded from {path}")

    def get_vae_metrics(self) -> Dict[str, float]:
        """Get recent VAE training metrics"""
        if not any(self.vae_losses.values()):
            return {}
        
        return {
            f'vae/{key}_recent': np.mean(values[-100:]) if values else 0.0
            for key, values in self.vae_losses.items()
        }

    def eval_mode(self):
        """Set agent to evaluation mode"""
        super().eval_mode()
        # Set VAE to eval mode
        self.vae.encoder.eval()
        self.vae.decoder0.eval()
        self.vae.decoder1.eval()
        self.vae.prior.eval()
        
    def evaluate_vae(
        self,
        vae: CLASVAE,
        val_buffer: VAEBuf,
        num_eval_batches: int = 50
    ):
        """
        Evaluate the VAE on held‐out data by re‐using vae.vae_loss, so that
        recon_loss/kl_loss match exactly the training objective.

        Returns:
            {
                'val_recon_loss': <average negative log-likelihood>,
                'val_kl':         <average KL>,
                'val_mse':        <optional average MSE>
            }
        """
        # Accumulators
        total_recon_ll = 0.0   # sum of recon_loss (negative log‐likelihood)
        total_kl       = 0.0
        total_mse      = 0.0   # optional: sum of plain MSE

        device = vae.device  # make sure we push tensors to the same device

        with torch.no_grad():
            for _ in range(num_eval_batches):
                # 1) Sample a batch of obs_dicts + actions from the validation buffer
                obs_dicts, actions_np = val_buffer.sample_batch(self.vae_batch_size)

                # 2) Convert obs_dicts → batched_obs (numpy → torch)  
                #    Note: _batch_observations returns a dict of (B, D) numpy arrays.
                batched_obs = self._batch_observations(obs_dicts)

                # 3) Move actions onto the correct device as a torch.Tensor
                actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)

                # 4) Compute the exact same losses used during training:
                #    recon_loss = −E[log p(u|o,v)]  (in “nats” units)
                #    kl_loss    = KL(q(v|o,u) || p(v|o))
                losses = vae.vae_loss(batched_obs, actions)
                total_recon_ll += losses['recon_loss'].item()
                total_kl       += losses['kl_loss'].item()

                # ------------------------------------------------------------
                # 5) OPTIONAL: compute a plain MSE between decoded actions and ground truth
                # ------------------------------------------------------------
                #    This requires re‐running encode→reparameterize→decode, exactly like in training.
                _, _, _, full_obs = vae.parse_observation(batched_obs)
                full_obs = full_obs.to(device)

                #    (a) encode & reparameterize
                mu_enc, logvar_enc = vae.encoder(full_obs, actions)
                latent_pre_tanh = vae.reparameterize(mu_enc, logvar_enc)
                latent_action   = torch.tanh(latent_pre_tanh)

                #    (b) decode to per-robot actions
                r0, r1 = vae.decode_actions(batched_obs, latent_action)
                recon_actions = torch.cat([r0, r1], dim=-1)  # shape: (B, 12)

                #    (c) mean‐squared‐error in joint‐velocity space
                mse = torch.mean((recon_actions - actions)**2)
                total_mse += mse.item()

        # Average over all batches
        avg_recon = total_recon_ll / num_eval_batches
        avg_kl    = total_kl       / num_eval_batches
        avg_mse   = total_mse      / num_eval_batches  # optional

        return {
            'val_recon_loss': avg_recon,  # negative log‐likelihood (nats)
            'val_kl':         avg_kl,     # KL term (nats)
            'val_mse':        avg_mse     # optional MSE
        }

    def train_mode(self):
        """Set agent to training mode"""
        super().train_mode()
        # Set VAE to train mode
        self.vae.encoder.train()
        self.vae.decoder0.train()
        self.vae.decoder1.train()
        self.vae.prior.train()