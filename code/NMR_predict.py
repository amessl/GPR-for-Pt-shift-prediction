
# soap_gpr.py
import numpy as np
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation, RBF
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, cross_val_score, KFold, \
    ShuffleSplit, RandomizedSearchCV, learning_curve
from sklearn.metrics import r2_score
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from generate_descriptors import generate_descriptors

class GPR_NMR(generate_descriptors):
    def __init__(self, descriptor_params, descriptor_path, central_atom, xyz_path, xyz_base,
                 descriptor_type, mode):

        """
       Initialize class for using descriptors as input for kernel regression (GPR and KRR)
       with cross-validated errors, hyperparameter tuning and visualization of learning curves

       :param mode: 'read' for passing pre-generated descriptors as input
       :param descriptor_type: Options currently implemented: 'SOAP' or 'APE-RF'
       :param descriptor_path: path where the pre-generated descriptors are stored
       :param descriptor_params: Parameters of the descriptors (SOAP: [rcut, nmax, lmax], APE-RF: [qmol, rcut, dim])
       :param central_atom: Atom symbol (str) of central atom ('Pt' for 195Pt-NMR)
       :param xyz_path: Path to directory where xyz-files are stored
       :param xyz_base: basename of the xyz_files (e.g. for st_1.xyz: 'st_')
       :param regressor_type: Whether to use 'GPR' or 'KRR' for regression
       """

        # use mode read or write, drop normalize and include it in predict function, drop descriptor
        # type and make the read/write function more general without having to specify descriptor type
        # Check generation of APE-RF descriptor (directory, storing, etc.)
        # Plotting learning curves, etc.: call predict function inside lc_function
        # Later: write code for using descriptors in non-local mode for other properties

        super().__init__(descriptor_params, descriptor_path, central_atom, xyz_path, xyz_base)
        self.mode = mode
        self.descriptor_type = descriptor_type

    def read_descriptors(self):

        dataset = []

        descriptor_path = os.path.join(self.descriptor_path,
                                       f'{self.descriptor_params[0]}_{self.descriptor_params[1]}_{self.descriptor_params[2]}/')

        descriptor_filenames = sorted(os.listdir(descriptor_path), key=lambda x: int(x.split('.')[0]))

        memory = 0
        file_count = 0

        for filename in descriptor_filenames:
            try:
                descriptor_file = os.path.join(descriptor_path, filename)
                descriptor_array = np.load(descriptor_file, allow_pickle=True)
                descriptor_array = descriptor_array.flatten()

                dataset.append(descriptor_array)
                memory += os.path.getsize(descriptor_file)

                file_count += 1

            except os.path.getsize(descriptor_file) == 0:
                raise Warning(f'File No. {file_count} is empty.')

                pass

        print(
            f'SOAP files read: {len(descriptor_filenames)} \nAverage size: {round((memory / file_count) / 1024, 3)} kB')

        return dataset


    def predict(self, regressor_type, kernel_degree, target_path, target_name, normalize, alpha,
                lc=None, correlation_plot=None, hypers=None, grid_search=None):

        if self.mode == 'read':

            X_data = self.read_descriptors()

        elif self.mode == 'write':

            if self.descriptor_type == 'SOAP':

                X_data = self.generate_SOAPs()

            elif self.descriptor_type == 'APE_RF':

                X_data = self.get_APE_RF(format='xyz')

            else:
                raise Exception('Descriptor type has to be specified. Use "SOAP" or "APE-RF"')

        else:
            raise Exception('Mode has to be specified as "read" for using pre-generated descriptors \n'
                            'or "write" for generating new descriptors and passing them as input"')

        if normalize:
            X_data = Normalizer(norm='l2').fit_transform(X_data)

        target_data = pd.read_csv(f'{target_path}.csv')[str(target_name)]

        randomSeed = 42
        train_X, test_X, train_target, test_target = train_test_split(X_data, target_data, random_state=randomSeed, test_size=0.25, shuffle=True)

        if regressor_type == 'GPR':
            if kernel_degree == 1:
                estimator = GaussianProcessRegressor(kernel=DotProduct(), random_state=randomSeed, alpha=float(alpha), optimizer=None)
            elif kernel_degree > 1:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(), int(kernel_degree)), random_state=randomSeed, alpha=float(alpha), optimizer=None)
            else:
                estimator = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1e-12,1e3)), alpha=float(alpha))
        elif regressor_type == 'KRR':
            if kernel_degree == 1:
                estimator = Ridge(alpha=float(alpha))
            elif kernel_degree > 1:
                estimator = KernelRidge(kernel=Exponentiation(DotProduct(), int(kernel_degree)), alpha=float(alpha))
            else:
                estimator = KernelRidge(kernel='rbf', alpha=float(alpha))
        else:
            raise ValueError('Regressor type has to be specified as "GPR" or "KRR".')

        estimator.fit(train_X, train_target)

        cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)

        scores_rmse = cross_val_score(estimator, X_data, target_data, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
        print('RMSE (4-fold CV):', np.mean(np.abs(scores_rmse)), '[ppm]', np.std(np.abs(scores_rmse)), '[ppm] (STDEV)')

        scores_mae = cross_val_score(estimator, X_data, target_data, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        print('MAE (4-fold CV):', np.mean(np.abs(scores_mae)), '[ppm]', np.std(np.abs(scores_mae)), '[ppm] (STDEV)')

        if lc:
            self._plot_learning_curve(estimator, X_data, target_data, kernel_degree)

        if correlation_plot:
            self._plot_correlation(estimator, train_X, test_X, train_target, test_target)

        if hypers:
            self._plot_hyperparameters(estimator)

        if grid_search:
            self._perform_grid_search(estimator, train_X, train_target)

        return np.mean(np.abs(scores_mae)), np.std(np.abs(scores_mae)), np.mean(np.abs(scores_rmse)), np.std(np.abs(scores_rmse))

    def _plot_learning_curve(self, estimator, X_data, target_data, kernel_degree, regressor):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_data, target_data, train_sizes=np.linspace(0.25, 1.0, 5), cv=ShuffleSplit(n_splits=4, test_size=0.25, random_state=42), scoring='neg_mean_absolute_error')
        plt.figure()
        plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, -test_scores.mean(axis=1), 'o-', color='g', label='Cross-validation score')
        plt.title(f'Learning Curve ({regressor} with degree={kernel_degree})')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def _plot_correlation(self, estimator, train_X, test_X, train_target, test_target):
        estimator.fit(train_X, train_target)
        prediction = estimator.predict(test_X)
        correlation = r2_score(test_target, prediction)
        plt.figure()
        plt.scatter(test_target, prediction, edgecolors=(0, 0, 0))
        plt.plot([test_target.min(), test_target.max()], [test_target.min(), test_target.max()], 'k--', lw=4)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.title(f'Correlation plot (R^2 = {correlation:.2f})')
        plt.show()

    def _plot_hyperparameters(self, estimator):
        sigma_0_range = np.logspace(-5, 5, num=10)
        noise_level_range = np.logspace(-5, 5, num=10)
        sigma_0_grid, noise_level_grid = np.meshgrid(sigma_0_range, noise_level_range)
        log_marginal_likelihood = [estimator.log_marginal_likelihood(theta=np.log([0.36, sigma_value, noise]))
                                   for sigma_value, noise in zip(sigma_0_grid.ravel(), noise_level_grid.ravel())]
        log_marginal_likelihood = np.reshape(log_marginal_likelihood, newshape=noise_level_grid.shape)
        vmin, vmax = (-log_marginal_likelihood).min(), 10
        level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=10), decimals=1)
        plt.contour(sigma_0_grid, noise_level_grid, -log_marginal_likelihood, levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Sigma 0")
        plt.ylabel("Noise-level")
        plt.title("Log-marginal-likelihood")
        plt.show()

    def _perform_grid_search(self, estimator, train_X, train_target):
        param_grid = {'alpha': [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]}
        cv = KFold(n_splits=4, shuffle=True, random_state=42)
        grid_search = RandomizedSearchCV(estimator, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, n_iter=100)
        grid_search.fit(train_X, train_target)
        best_params = grid_search.best_params_
        best_mae = -grid_search.best_score_
        print('Best params:', best_params)
        print('Best MAE:', best_mae)


# TODO: Refactor predict function
# TODO: Include option for noise_estimation using WhiteKernel
# TODO: add function for single predictions