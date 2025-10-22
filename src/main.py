
import hydra
from omegaconf import DictConfig
from src.predict_sklearn import SklearnGPRegressor

# TODO: Save fits and descriptors to separate directory when training on full set (partitioned=False)

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def eval_model(cfg: DictConfig):
    """
        Execute the end-to-end Gaussian Process Regression (GPR) pipeline using the default configs
        managed by hydra.

        This function acts as the main entry point for orchestrating the model workflow,
        including training, testing on holdout set, and sequential execution, depending on the configured task.
        Hydra for hierarchical configuration management to ensure reproducible runs.

        Parameters
        ----------
        cfg : omegaconf.DictConfig
            Hydra configuration object containing all experiment parameters, including
            backend settings, task mode, and reporting options.

        Raises
        ------
        ValueError
            If an invalid task mode is specified in the configuration file.

        Notes
        -----
        The pipeline supports three modes:
            - 'train': Perform model fitting using training data and specified hyperparameters.
            - 'test': Evaluate the trained model on test data and generate performance reports.
            - 'null': Sequentially execute both training and testing stages.

        The trained model and evaluation artifacts are persisted based on the reporting configuration.

    """

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
