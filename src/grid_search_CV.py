import os
import numpy.linalg
from NMR_predict import GPR_NMR
import itertools
from multiprocessing import Pool
import shutil


def eval_APE_RF_hyperparams(hyperparams, paths, noise_estim=True):

    """
    Evaluates cross-validated mean absolute error (MAE) for a given hyperparameter combination (APE-RF)

    :param hyperparams: List of hyperparameters
    :param paths: List of paths to representations, structures (xyz-files) and target data
    :return: List of hyperparameters and corresponding MAE
    """

    rcut, dim, noise, kernel_degree = hyperparams
    descriptor_paths = paths[:2]
    xyz_paths = paths[2:4]
    target_paths = paths[4:6]

    model = GPR_NMR(descriptor_params=[rcut, dim],
                    descriptor_path=descriptor_paths,
                    central_atom='Pt',
                    xyz_path=xyz_paths, xyz_base='st_',
                    descriptor_type='APE-RF', mode='write',
                    target_path=target_paths)

    try:
        errors_std = model.GPR_train(kernel_degree=kernel_degree, noise=noise, noise_estim=noise_estim)  # sklearn backend

        if noise_estim:
            ls_hyperparams = list(hyperparams)
            ls_hyperparams.append(errors_std[4])

            hyperparams = tuple(ls_hyperparams)

    except numpy.linalg.LinAlgError:
        print(f'Parameter combination {hyperparams} '
              f'produced singular kernel matrix. Proceeding iteration.')

        errors_std = [float('inf')]
        pass

    return hyperparams, errors_std[0]


def tune_APE_RF_hyperparams(rcut_grid, dim_grid, noise_grid, kernel_grid, paths, n_procs, burn=False):

    """
    Parallel exhaustive grid search iterating over all hyperparameter combinations specified
    by the corresponding grids to find the combination yielding the minimal MAE (for APE-RF)

    :param rcut_grid: List of cutoff radii to iterate over
    :param dim_grid: List of feature dimensions to iterate over
    :param noise_grid: List of noise values to iterate over
    :param kernel_grid: List of polynomial kernel degrees to iterate over
    :param paths: List of paths
    :param n_procs: Number of parallel processes
    :param burn: Whether to keep all representation files or remove them after the optiimization
    :return: Parameter combination yielding the minimal MAE and the MAE
    """

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

def eval_SOAP_hyperparams(hyperparams, paths, noise_estim=True):

    """
     Evaluates cross-validated mean absolute error (MAE) for a given hyperparameter combination (SOAP)

     :param hyperparams: List of hyperparameters
     :param paths: List of paths to representations, structures (xyz-files) and target data
     :return: List of hyperparameters and corresponding MAE
     """

    rcut, nmax, lmax, noise, kernel_degree = hyperparams
    descriptor_paths = paths[:2]
    xyz_paths = paths[2:4]
    target_paths = paths[4:6]

    model = GPR_NMR(descriptor_params=[rcut, nmax, lmax],
                    descriptor_path=descriptor_paths,
                    central_atom='Pt',
                    xyz_path=xyz_paths, xyz_base='st_',
                    descriptor_type='SOAP', mode='write', target_path=target_paths)

    try:
        errors_std = model.GPR_train(kernel_degree=kernel_degree, noise=noise, noise_estim=noise_estim) # sklearn backend

        if noise_estim:
            ls_hyperparams = list(hyperparams)
            ls_hyperparams.append(errors_std[4])

            hyperparams = tuple(ls_hyperparams)

    except numpy.linalg.LinAlgError:
        print(f'Parameter combination {hyperparams} '
              f'produced singular kernel matrix. Proceeding iteration.')

        errors_std = [float('inf')]
        pass

    return hyperparams, errors_std[0]

def tune_SOAP_hyperparams(rcut_grid, nmax_grid, lmax_grid, noise_grid, kernel_grid, paths, n_procs, burn=True):

    """
        Parallel exhaustive grid search iterating over all hyperparameter combinations specified
        by the corresponding grids to find the combination yielding the minimal MAE (for SOAP)

        :param rcut_grid: List of cutoff radii to iterate over
        :param nmax_grid: List of max number of radial basis functions to iterate over
        :param lmax_grid: List of maximum degree of spherical harmonics to iterate over
        :param kernel_grid: List of polynomial kernel degrees to iterate over
        :param paths: List of paths
        :param n_procs: Number of parallel processes
        :param burn: Whether to keep all representation files or remove them after the optiimization
        :return: Parameter combination yielding the minimal MAE and the MAE
        """

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
        print('Errors of hyperparameter combination:', params, mae) # TODO: iterate total MAE list for lowest and top candidates

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

        print('Representation files removed. To keep them set burn=False')

    return best_params, best_mae

