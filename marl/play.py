""" Script to play a trained model. """

import os
import argparse
from omegaconf import OmegaConf

from marl.utils.config_utils import instantiate_all
from marl.utils.utils import set_seed

def play(checkpoint: str, record: bool):
    """Load a policy from a checkpoint and play it in an environment.
    
    Provide the experiment path .../{date}
    """
    config_path = os.path.join(checkpoint, ".hydra/config.yaml")
    model_path = os.path.join(checkpoint, "policy")
    config = OmegaConf.load(config_path)

    if record:
        config.record = True

    config.environment.num_envs = 1

    set_seed(config.seed)    
    env, _, _, agent = instantiate_all(config)

    obs,_ = env.reset()
    agent.load(model_path)

    done, truncated = False, False
    while not done or not truncated:
        action = agent.act_inference(obs)
        obs, reward, done, truncated, info = env.step(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--record", type=bool, help="Record videos during rollout", default=False)

    args = parser.parse_args()
    play(args.checkpoint, args.record)