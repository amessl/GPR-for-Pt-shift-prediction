import os
import numpy as np
import numpy.linalg
from predict_sklearn import SklearnGPRegressor
import itertools
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import subprocess
import webbrowser
import time


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def eval_model_grid(cfg: DictConfig, mlflow_ui: bool = True):

    gp_grid = OmegaConf.to_container(cfg.grid_search.GP_grid, resolve=True)

    if cfg.representations.rep == 'SOAP':
        param_grid = OmegaConf.to_container(cfg.grid_search.SOAP_grid, resolve=True)
        param_names = list(cfg.grid_search.SOAP_grid.keys())

    elif cfg.representations.rep == 'GAPE':
        param_grid = OmegaConf.to_container(cfg.grid_search.GAPE_grid, resolve=True)
        print('Param Grid GAPE:', param_grid)
        param_names = list(cfg.grid_search.GAPE_grid.keys())

    else:
        raise ValueError(f"Representation {cfg.representations.rep} not supported. Use 'GAPE' or 'SOAP'.")

    combined_grid = {**gp_grid, **param_grid}
    param_names = list(gp_grid.keys()) + param_names

    grid = list(itertools.product(*combined_grid.values()))

    for iteration, hyperparams in enumerate(grid, 1):
        hyperparams_dict = dict(zip(param_names, hyperparams))

        # Clone cfg so we don't overwrite in-place
        updated_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        # Update GP hyperparams
        for key in gp_grid.keys():
            if key in hyperparams_dict:
                updated_cfg.backend.training[key] = hyperparams_dict[key]

        # Update representation-specific hyperparams
        if cfg.representations.rep == 'SOAP':
            for key in param_grid.keys():
                updated_cfg.representations.SOAP_params[key] = hyperparams_dict[key]

        elif cfg.representations.rep == 'GAPE':
            for key in param_grid.keys():
                updated_cfg.representations.GAPE_params[key] = hyperparams_dict[key]

        print('-' * 35)
        print(f'\nIteration {iteration}/{len(grid)}:')
        print('-' * 35)
        print(f'Hyperparameters: {hyperparams}')
        print(f'{hyperparams_dict}')

        eval_param_comb(hyperparams_dict, updated_cfg)

    if mlflow_ui:

        mlflow_URI = mlflow.get_tracking_uri()[7:]
        print(mlflow_URI)
        subprocess.Popen(["mlflow", "ui", "--backend-store-uri", mlflow_URI])

        time.sleep(2)
        webbrowser.open("http://127.0.0.1:5000")

def eval_param_comb(hyperparams_dict: dict, cfg: DictConfig):

    # TODO: Update the hyperparams in the config (otherwise you train same model over and over again in grid search)

    print("MLflow tracking URI:", mlflow.get_tracking_uri()[7:])

    with mlflow.start_run():
        mlflow.log_params(hyperparams_dict)

        model = SklearnGPRegressor(config=cfg)

        try:
            train_MAE = model.gpr_train(**cfg.backend.training)[0]
            mlflow.log_metric("train_mae", train_MAE)

        except np.linalg.LinAlgError:
            print(f"Combination {hyperparams_dict} produced singular kernel matrix.")
            mlflow.log_metric("train_mae", float("inf"))

        except Exception as e:
            print(f"Run failed for {hyperparams_dict}: {e}")
            mlflow.log_metric("train_mae", float("nan"))

        finally:
            print(f"Logged run for: {hyperparams_dict}")

if __name__ == "__main__":
    hydra.main(config_path="../conf", config_name="config", version_base="1.1")(eval_model_grid)()
