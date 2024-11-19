import os
from NMR_predict import GPR_NMR
import itertools
from multiprocessing import Pool
import shutil


def eval_APE_RF_hyperparams(hyperparams, paths):

    rcut, dim, noise, kernel_degree = hyperparams
    descriptor_paths = paths[:2]
    xyz_paths = paths[2:4]
    target_paths = paths[4:6]

    model = GPR_NMR(descriptor_params=[rcut, dim],
                    descriptor_path=descriptor_paths,
                    central_atom='Pt',
                    xyz_path=xyz_paths, xyz_base='st_',
                    descriptor_type='APE_RF', mode='write')

    errors_std = model.GPR_predict(kernel_degree=kernel_degree,
                                   target_path=target_paths,
                                   target_name='Experimental',
                                   normalize=False, noise=noise,
                                   partitioned=True)

    return hyperparams, errors_std[0]


def tune_APE_RF_hyperparams(rcut_grid, dim_grid, noise_grid, kernel_grid, paths, n_procs, burn=False):

    param_grid = {
        'rcut': rcut_grid,
        'dim': dim_grid,
        'noise': noise_grid,
        'degree': kernel_grid
    }
    param_combinations = list(itertools.product(*param_grid.values()))
    print(f'Number of parameter combinations: {len(param_combinations)}')

    min_diff = 1e-2
    best_mae = float('inf')
    best_params = None
    top_candidates = []

    with Pool(processes=n_procs) as pool:
        results = pool.starmap(eval_APE_RF_hyperparams, [(params, paths) for params in param_combinations])

    for params, mae in results:
        print('Errors of hyperparameter combination:', params, mae)

        if mae < best_mae:
            best_mae = mae
            best_params = dict(zip(param_grid.keys(), params))
            top_candidates = [best_params]

        elif abs(mae - best_mae) <= min_diff:
            top_candidates.append(dict(zip(param_grid.keys(), params)))

    print(f"Optimized hyperparameters: {best_params}")
    print(f"Lowest cross-validated MAE: {best_mae}")
    print("Other top hyperparameter combinations within MAE tolerance:", top_candidates)

    if burn:
        for params_list in param_combinations:
            for path in paths[:2]:
                descriptor_folder = os.path.join(path, f'{str(params_list[0])}_{str(params_list[1])}')
                if os.path.exists(descriptor_folder):
                    shutil.rmtree(descriptor_folder)
                else:
                    pass

    return best_params, best_mae

def eval_SOAP_hyperparams(hyperparams, paths):
    rcut, nmax, lmax, noise, kernel_degree = hyperparams
    descriptor_paths = paths[:2]
    xyz_paths = paths[2:4]
    target_paths = paths[4:6]

    model = GPR_NMR(descriptor_params=[rcut, nmax, lmax],
                    descriptor_path=descriptor_paths,
                    central_atom='Pt',
                    xyz_path=xyz_paths, xyz_base='st_',
                    descriptor_type='SOAP', mode='write')

    errors_std = model.GPR_predict(kernel_degree=kernel_degree,
                                   target_path=target_paths,
                                   target_name='Experimental',
                                   normalize=True, noise=noise,
                                   partitioned=True)

    return hyperparams, errors_std[0]

def tune_SOAP_hyperparams(rcut_grid, nmax_grid, lmax_grid, noise_grid, kernel_grid, paths, n_procs, burn=True):
    param_grid = {
        'rcut': rcut_grid,
        'nmax': nmax_grid,
        'lmax': lmax_grid,
        'noise': noise_grid,
        'degree': kernel_grid
    }
    param_combinations = list(itertools.product(*param_grid.values()))
    print(f'Number of parameter combinations: {len(param_combinations)}')

    min_diff = 1e-2
    best_mae = float('inf')
    best_params = None
    top_candidates = []

    with Pool(processes=n_procs) as pool:
        results = pool.starmap(eval_SOAP_hyperparams, [(params, paths) for params in param_combinations])

    for params, mae in results:
        print('Errors of hyperparameter combination:', params, mae)

        if mae < best_mae:
            best_mae = mae
            best_params = dict(zip(param_grid.keys(), params))
            top_candidates = [best_params]
        elif abs(mae - best_mae) <= min_diff:
            top_candidates.append(dict(zip(param_grid.keys(), params)))

    print(f"Optimized hyperparameters: {best_params}")
    print(f"Lowest cross-validated MAE: {best_mae}")
    print("Other top hyperparameter combinations within MAE tolerance:", top_candidates)

    if burn:
        for params_list in param_combinations:
            for path in paths[:2]:
                descriptor_folder = os.path.join(path, f'{str(params_list[0])}_{str(params_list[1])}_{str(params_list[2])}')
                if os.path.exists(descriptor_folder):
                    shutil.rmtree(descriptor_folder)
                else:
                    pass

    # TODO: Update APE_RF generation and grid_search with all features (parallelization with joblib, burn, etc.)

        print('Representation files removed. To keep them set burn=False')

    return best_params, best_mae

