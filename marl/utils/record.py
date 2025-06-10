import os
# MUST be set BEFORE importing anything else that uses OpenGL/MuJoCo
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from marl.utils.config_utils import instantiate_all
from marl.algorithms.sac import CLASSAC
from marl.networks.CLAS_vae import CLASVAE
import logging
import imageio
import os
from tqdm import tqdm
import robosuite.macros as macros
import matplotlib

matplotlib.use('Agg')

# Set the image convention to opencv so that the images are automatically rendered "right side up"
macros.IMAGE_CONVENTION = "opencv"

@hydra.main(version_base=None, config_path="../.configs", config_name="config")
def evaluate_and_record(cfg: DictConfig):
    """Evaluate trained policy and record video"""
    
    print("Loading environment and models...", flush=True)
    
    # Instantiate environment with video recording setup
    env, policy, algorithm, agent = instantiate_all(cfg)
    
    # Load VAE model
    vae_config = cfg.agent.clas_training.vae_config
    vae = CLASVAE(config=dict(vae_config))
    
    # Load VAE weights - Use correct filename
    vae_checkpoint = torch.load("clas_vae_prefilled_beta.pt", map_location=vae.device, weights_only=False)
    vae.encoder.load_state_dict(vae_checkpoint['encoder'])
    vae.decoder0.load_state_dict(vae_checkpoint['decoder0'])
    vae.decoder1.load_state_dict(vae_checkpoint['decoder1'])
    vae.prior.load_state_dict(vae_checkpoint['prior'])
    vae.eval()
    
    print("VAE model loaded successfully!")
    
    # Initialize SAC agent
    sac_config = cfg.agent.clas_training.sac_config
    obs_dim = vae.full_obs_dim
    latent_dim = vae.latent_dim
    
    sac_agent = CLASSAC(vae, obs_dim, latent_dim, dict(sac_config))
    
    # Load SAC weights with correct keys
    print("Loading SAC checkpoint...")
    ckpt = torch.load("sac_weights/clas_sac_ep00500.pt", map_location=sac_agent.device, weights_only=False)
    
    # Use the correct key names from your save_checkpoint method
    sac_agent.policy.load_state_dict(ckpt['policy_state'])
    sac_agent.q1.load_state_dict(ckpt['q1_state'])
    sac_agent.q2.load_state_dict(ckpt['q2_state'])
    sac_agent.q1_target.load_state_dict(ckpt['q1_target'])
    sac_agent.q2_target.load_state_dict(ckpt['q2_target'])
    sac_agent.log_alpha.data.copy_(ckpt['log_alpha'].to(sac_agent.device))
    
    episode = ckpt.get('episode', 500)
    avg_reward = ckpt.get('avg_reward', 0.0)
    print(f"Loaded SAC model from episode {episode} with avg reward {avg_reward:.2f}")
    
    # Set to evaluation mode
    sac_agent.policy.eval()
    sac_agent.q1.eval()
    sac_agent.q2.eval()
    
    # Create output directory
    os.makedirs("evaluation_videos", exist_ok=True)
    
    # Evaluation parameters
    num_episodes = 3
    max_episode_steps = 500
    fps = 20
    skip_frame = 5  # Record every 5th frame to reduce file size
    
    for episode_idx in range(num_episodes):
        print(f"\nEvaluating Episode {episode_idx + 1}/{num_episodes}")
        
        # Setup video writer using imageio (like robosuite example)
        video_path = f"evaluation_videos/clas_policy_ep{episode_idx:03d}.mp4"
        writer = imageio.get_writer(video_path, fps=fps)
        
        # Reset environment
        obs_dict, _ = env.reset()
        episode_reward = 0.0  # Initialize as float
        step_count = 0
        
        # Episode loop
        pbar = tqdm(total=max_episode_steps, desc=f"Episode {episode_idx+1}", unit="step")
        
        done = False
        while not done and step_count < max_episode_steps:
            
            # Get action from policy
            if hasattr(env, 'num_envs') and env.num_envs > 1:
                # Vectorized environment
                batched_obs = sac_agent._batch_vectorized_observations(obs_dict, env.num_envs)
                actions_batch, latent_actions_batch = sac_agent.select_action_batch(batched_obs, evaluate=True)
                action = actions_batch[0]  # Take first environment
            else:
                # Single environment
                action, latent_action = sac_agent.select_action(obs_dict, evaluate=True)
            
            # Environment step
            if hasattr(env, 'num_envs') and env.num_envs > 1:
                next_obs_dict, rewards, dones, _, _ = env.step(actions_batch)
                reward = float(rewards[0])  # Convert to float scalar
                done = bool(dones[0])      # Convert to bool scalar
                obs_dict = {k: v[0] for k, v in next_obs_dict.items()}  # Extract first env
            else:
                next_obs_dict, reward, done, truncated, info = env.step(action)
                reward = float(reward) if hasattr(reward, 'item') else float(reward)  # Ensure scalar
                done = bool(done or truncated)
                obs_dict = next_obs_dict
            
            # Capture frame for video (every skip_frame steps)
            if step_count % skip_frame == 0:
                try:
                    frame = None
                    
                    # Try to get camera image from observation
                    if 'agentview_image' in obs_dict:
                        frame = obs_dict['agentview_image']
                    elif 'frontview_image' in obs_dict:
                        frame = obs_dict['frontview_image']
                    else:
                        # Fallback: render environment
                        rendered = env.render()
                        if isinstance(rendered, tuple):
                            frame = rendered[0]  # Take first element if tuple
                        else:
                            frame = rendered
                    
                    if frame is not None and hasattr(frame, 'dtype'):
                        # Convert to uint8 if needed
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                        writer.append_data(frame)
                    else:
                        print(f"Warning: Invalid frame at step {step_count}: {type(frame)}")
                    
                except Exception as e:
                    print(f"Warning: Could not capture frame at step {step_count}: {e}")
            
            episode_reward += reward
            step_count += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Step': step_count,
                'Action_norm': f'{np.linalg.norm(action):.3f}'
            })
        
        pbar.close()
        
        # Close video writer
        writer.close()
        print(f"Video saved: {video_path}")
        
        print(f"Episode {episode_idx + 1} completed:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Success: {'Yes' if episode_reward > 100 else 'Maybe' if episode_reward > 50 else 'No'}")
    
    env.close()
    print(f"\nEvaluation completed! {num_episodes} videos saved in 'evaluation_videos/' directory")

if __name__ == "__main__":
    # Suppress robosuite logging
    logging.getLogger("robosuite").setLevel(logging.ERROR)
    logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
    
    evaluate_and_record()