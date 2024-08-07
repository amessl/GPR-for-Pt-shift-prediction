
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

class GPR_NMR:
    def __init__(self, mode, descriptor_path, descriptor_type, descriptor_params, regressor_type, normalize):

        """
               Initialize class for using descriptors as input for kernel regression (GPR and KRR)
               with cross-validated errors, hyperparameter tuning and visualization of learning curves

               :param mode: either 'read' for directly using pre-generated descriptors or
                'write' to generate descriptors and pass them as input for the regression models
               :param descriptor_path: path where the pre-generated descriptors are stored
               :param descriptor_type: 'SOAP' or 'APE_RF'
               :param descriptor_params: Parameters of the descriptors (SOAP: [rcut, nmax, lmax], APE-RF: [qmol, rcut, dim])
               :param regressor_type: Whether to use 'GPR' or 'KRR' for regression
               :param normalize: Whether to normalize the feature vectors before passing them as input for regression
               """

        self.mode = mode
        self.descriptor_path = descriptor_path
        self.descriptor_type = descriptor_type
        self.descriptor_params = descriptor_params
        self.regressor_type = regressor_type
        self.normalize = normalize


# TODO: finish read_descriptors with correct path and refactor predict function
    def read_descriptors(self):

        dataset = []

        if self.descriptor_type == 'APE_RF':

            APE_RF_path = os.path.join(self.descriptor_path,
                                           f'qmol{self.descriptor_params[0]}_rcut{self.descriptor_params[1]}_dim{self.descriptor_params[2]}/')

            APE_RF_filenames = sorted(os.listdir(APE_RF_path), key=lambda x: int(x.split('.')[0]))

            file_count = 0

            for filename in APE_RF_filenames:
                try:
                    file = os.path.join(APE_RF_path, filename)
                    APE_RF_array = np.loadtxt(file)
                    dataset.append(APE_RF_array)

                    file_count += 1

                except os.path.getsize(file) == 0:
                    raise Warning(f'File No. {file_count} is empty.')

                    pass

            print(
                f'SOAP files read: {len(APE_RF_filenames)}')

        elif self.descriptor_type == 'SOAP':

            SOAP_path = os.path.join(self.descriptor_path,
                                          f'r{self.descriptor_params[0]}_n{self.descriptor_params[1]}_l{self.descriptor_params[2]}/')


            SOAP_filenames = sorted(os.listdir(SOAP_path), key=lambda x: int(x.split('.')[0]))
            SOAP_memory = 0
            file_count = 0

            for SOAP_filename in SOAP_filenames:
                try:
                    SOAP_file = os.path.join(SOAP_path, SOAP_filename)
                    SOAP_array = np.load(SOAP_file)
                    dataset.append(SOAP_array)
                    SOAP_memory += os.path.getsize(SOAP_file)

                    file_count += 1

                except os.path.getsize(SOAP_file) == 0:
                    raise Warning(f'File No. {file_count} is empty.')

                    pass

            print(
                f'SOAP files read: {len(SOAP_filenames)} \nAverage size: {round((SOAP_memory / file_count) / 1024, 3)} kB')

            return dataset


    def predict(self, kernel_degree, target_path, target_name, alpha,
                lc=None, correlation_plot=None, hypers=None, grid_search=None):

        if self.mode == 'read':
            X_data = self.read_descriptors()
        elif self.mode == 'write':
            X_data = self.generate_descriptors()
        else:
            raise ValueError('Mode has to be specified as "read" or "write".')

        target_data = pd.read_csv(f'{target_path}.csv')[str(target_name)]

        randomSeed = 42
        train_X, test_X, train_target, test_target = train_test_split(X_data, target_data, random_state=randomSeed, test_size=0.25, shuffle=True)

        if self.regressor_type == 'GPR':
            if kernel_degree == 1:
                estimator = GaussianProcessRegressor(kernel=DotProduct(), random_state=randomSeed, alpha=float(alpha), optimizer=None)
            elif kernel_degree > 1:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(), int(kernel_degree)), random_state=randomSeed, alpha=float(alpha), optimizer=None)
            else:
                estimator = GaussianProcessRegressor(kernel=RBF(), alpha=float(alpha))
        elif self.regressor_type == 'KRR':
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
            self._plot_learning_curve(estimator, X_data, target_data, kernel_degree, regressor)

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
