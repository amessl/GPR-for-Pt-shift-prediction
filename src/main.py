
import os
import hydra
from omegaconf import DictConfig
from src.predict_sklearn import SklearnGPRegressor

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def eval_model(cfg: DictConfig):

    cfg.representations.rep = 'ChEAP'
    zeta = 2

    cfg.backend.training.kernel_degree = zeta
    cfg.backend.testing.kernel_degree = zeta

    model = SklearnGPRegressor(config=cfg)
    model.gpr_train(**cfg.backend.training, report='full')

    model.gpr_test(**cfg.backend.testing, report='full')

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    eval_model()
