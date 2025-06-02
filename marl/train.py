""" Script to train a MARL agent with the provided configuration file."""

from omegaconf import DictConfig

import hydra
from marl.utils.config_utils import instantiate_all
from marl.utils.utils import set_seed
from hydra.core.hydra_config import HydraConfig
import os

@hydra.main(version_base=None, config_path=".configs", config_name="config")
def train(config: DictConfig):
    """Train a MARL agent with the provided configuration file."""
    save_dir = HydraConfig.get().runtime.output_dir

    set_seed(config.seed)
    env, _, _, agent = instantiate_all(config)

    env.reset()

    try:
        agent.learn(config.total_timesteps)

    except KeyboardInterrupt:
        print("Keyboard interrupt")
        agent.save(os.path.join(save_dir, "policy"))

    finally:
        env.close()
        agent.save(os.path.join(save_dir, "policy"))

if __name__ == "__main__":
    train()