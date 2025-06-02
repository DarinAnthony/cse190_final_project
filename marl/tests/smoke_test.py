import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../.configs", config_name="config")
def hydra_smoke(cfg: DictConfig):
    print("HYDRA-SMOKE: I made it into the function!", flush=True)
    print("config was:", cfg, flush=True)

if __name__ == "__main__":
    hydra_smoke()
