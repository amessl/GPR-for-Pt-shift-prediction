
import hydra
from omegaconf import DictConfig
from src.predict_sklearn import SklearnGPRegressor

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def eval_model(cfg: DictConfig):

    model = SklearnGPRegressor(config=cfg)

    if cfg.task == "train":
        model.gpr_train(**cfg.backend.training, report=cfg.report)

    elif cfg.task == "test":
        model.gpr_test(**cfg.backend.testing, report=cfg.report)

    elif cfg.task is None:
        print("No mode specified. Carrying out subsequent train and test runs")

        model.gpr_train(**cfg.backend.training, report=cfg.report)
        model.gpr_test(**cfg.backend.testing, report=cfg.report)

    else:
        raise ValueError("Invalid task. Use 'train', 'test' or None.")

if __name__ == "__main__":
    eval_model()
