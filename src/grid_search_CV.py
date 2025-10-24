import numpy as np
from predict_sklearn import SklearnGPRegressor
import itertools
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import subprocess
import webbrowser
import time
from multiprocessing import Pool

# Update this path to match your local env
MLFLOW_TRACKING_URI = '/home/alex/Pt_NMR/mlruns' # Change

def worker(args):
    """Worker function for parallel grid search execution.

    Parameters
    ----------
    args : tuple
        Tuple containing (hyperparams_dict, cfg_dict) where hyperparams_dict
        contains hyperparameter values and cfg_dict is the serialized configuration.
    """
    hyperparams_dict, cfg_dict = args
    cfg = OmegaConf.create(cfg_dict)
    eval_param_comb(hyperparams_dict, cfg)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def eval_model_grid(cfg: DictConfig, mlflow_ui: bool = True, parallel: bool = False):
    """Perform grid search over GP and descriptor hyperparameters.

    Executes an exhaustive grid search of descriptor-specific parameters (e.g., SOAP
    cutoff radius, nmax/lmax). For each iteration over descriptor parameters and kernel degree,
    the GP hyperparmaeters (noise variance and kernel bias) are optimized using gradient based (L-BFGS)
    maximization of the log marginal likelihood. Logs all runs to MLflow for comparing model performance
    in terms of the chosen error metric (MAE in this case).

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra configuration object containing:
        - cfg.grid_search.GP_grid : dict
            Grid of GP hyperparameters to search
        - cfg.grid_search.SOAP_grid or GAPE_grid : dict
            Grid of descriptor parameters
        - cfg.representations.rep : str
            Descriptor type ('SOAP' or 'GAPE')
        - cfg.experiment_name : str
            MLflow experiment name
    mlflow_ui : bool, optional
        If True, launch MLflow UI in browser. Default is True.
    parallel : bool, optional
        If True, execute grid search in parallel. Default is True.

    Raises
    ------
    ValueError
        If the representation type is not 'SOAP' or 'GAPE'.
    """

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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if parallel:
        tasks = []

        for hyperparams in grid:
            hyperparams_dict = dict(zip(param_names, hyperparams))
            updated_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

            for key in gp_grid.keys():
                if key in hyperparams_dict:
                    updated_cfg.backend.training[key] = hyperparams_dict[key]
            if cfg.representations.rep == 'SOAP':
                for key in param_grid.keys():
                    updated_cfg.representations.SOAP_params[key] = hyperparams_dict[key]
            elif cfg.representations.rep == 'GAPE':
                for key in param_grid.keys():
                    updated_cfg.representations.GAPE_params[key] = hyperparams_dict[key]

            # Store config as dict
            tasks.append((hyperparams_dict, OmegaConf.to_container(updated_cfg, resolve=True)))

        with Pool(processes=1) as pool:
            pool.map(worker, tasks)


    else:
        for iteration, hyperparams in enumerate(grid, 1):
            hyperparams_dict = dict(zip(param_names, hyperparams))

            # Clone cfg
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

        mlflow_URI = mlflow.get_tracking_uri()

        print(f'MLflow tracking URI: {mlflow_URI}')
        subprocess.Popen(["mlflow", "ui", "--backend-store-uri", mlflow_URI])

        time.sleep(2)
        webbrowser.open("http://127.0.0.1:5000")


def eval_param_comb(hyperparams_dict: dict, cfg: DictConfig):
    """Evaluate a single hyperparameter combination.

    Trains a GP model with the specified hyperparameters and logs results
    to MLflow. Handles failures gracefully by logging infinity or NaN.

    Parameters
    ----------
    hyperparams_dict : dict
        Dictionary of hyperparameter names and values.
    cfg : omegaconf.DictConfig
        Hydra configuration object.

    Notes
    -----
    Logs train MAE to mlflow. Failed runs log inf (singular kernel matrix) or
    NaN (other exceptions).
    """

    experiment_name = cfg.experiment_name
    mlflow.set_experiment(experiment_name)

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
