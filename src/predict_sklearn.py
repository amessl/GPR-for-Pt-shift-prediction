import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation, RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from base import BaseConfig
from omegaconf import OmegaConf
from data_loader import DataLoader

matplotlib.use("TkAgg")

class SklearnGPRegressor(BaseConfig):
    def __init__(self, config):
        """
       Initialize class for using descriptors as input for Gaussian Process Regression
       with cross-validated errors, hyperparameter tuning and visualization of learning curves

       """
        super().__init__(config)

        self.data_loader = DataLoader(config)
        self.mode = config.mode

        target_path = config.backend.model.target_path
        if isinstance(target_path, str):
            self.target_path = [target_path]
        elif OmegaConf.is_list(target_path):
            self.target_path = list(target_path)
        else:
            raise ValueError("Paths should be a string or a list of strings")

        self.fit_path = config.backend.model.fit_path
    def gpr_train(self, kernel_degree, noise, save_fit=True, stratify_train=True,
                  ard=False, report=None):

        """
        Uses the sklearn implementation of Gaussian Process Regression. Defines GPR model with linear/
        polynomial kernel and evaluates a given hyperparameter combination (of the representation and the GPR model)
        on a given dataset using k-fold cross-validation. Provides learning curves for the training and validation set
        and option of optimizing the noise level based on the gradient of the log marginal likelihood (LML) (sklearn backend)

        :param kernel_degree: Degree of the polynomial kernel
        :param noise: Likelihood variance a. k. a. noise level of the data (is only added to K(X,X) of training points,
        noise is added to K(X,X) of test points when using WhiteKernel()
        :param lc: Whether to generate and plot learning curve
        :param save_fit: Whether to save the state of the fitted model

        :return: CV MAE and RMSE and corresponding standard deviations
        """

        X_data = self.data_loader.load_samples()[0]
        target_data = self.data_loader.load_targets(target_name='Experimental')[0]

        randomSeed = 42

        if kernel_degree == 1:

            estimator = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(noise_level=noise),
                                                 random_state=randomSeed,
                                                 alpha=0.0, n_restarts_optimizer=10, normalize_y=True)

        elif kernel_degree > 1:

            estimator = GaussianProcessRegressor(
                kernel=Exponentiation(DotProduct(), int(kernel_degree)) + WhiteKernel(noise_level=noise),
                random_state=randomSeed,
                alpha=0.0, n_restarts_optimizer=10, normalize_y=True)  # always normalize labels !!

        else:

            estimator = GaussianProcessRegressor(kernel=ConstantKernel() * RBF() + WhiteKernel(noise_level=noise), alpha=0.0,
                                                 random_state=randomSeed, n_restarts_optimizer=10, normalize_y=True)

            if ard:

                X_data = StandardScaler().fit_transform(X_data)

                ard_kernel = ConstantKernel() * RBF(length_scale=np.ones(np.array(X_data).shape[1])) + WhiteKernel(noise_level=noise)

                estimator = GaussianProcessRegressor(kernel=ard_kernel, alpha=0.0,
                                                     random_state=randomSeed, n_restarts_optimizer=10, normalize_y=True)

        # TODO: Check vals of noise (normalize_y)
        print('Training in progress....')
        print(f'Init noise: {noise}')

        estimator.fit(X_data, target_data)

        opt_noise = estimator.kernel_.k2.noise_level
        print(f'\nOptimized noise variance: {opt_noise}')

        if kernel_degree == 1:
            opt_const = estimator.kernel_.k1.sigma_0
        else:
            opt_const = estimator.kernel_.k1.kernel.sigma_0

        print(f'Optimized kernel bias: {opt_const}')

        lml = estimator.log_marginal_likelihood_value_
        print(f'Log marginal likelihood: {lml} \n')

        if kernel_degree == 0:

            opt_lengthscale = estimator.kernel_.k1.k2.length_scale
            print(f'Optimized RBF lengthscale: {opt_lengthscale} \n')

        if stratify_train:

            strat_kf = StratifiedKFold(n_splits=4, random_state=randomSeed, shuffle=True)

            scores_mae = []
            scores_rmse = []

            bins = pd.qcut(target_data, q=10, labels=False, duplicates='drop')

            X_data = pd.DataFrame(X_data)
            target_data = pd.DataFrame(target_data)

            for fold, (train_index, val_index) in enumerate(strat_kf.split(X_data, bins)):
                X_train = X_data.iloc[train_index]
                target_train = target_data.iloc[train_index]

                X_val = X_data.iloc[val_index]
                target_val = target_data.iloc[val_index]

                estimator.fit(X_train, target_train)
                pred = estimator.predict(X_val)

                scores_mae.append(mean_absolute_error(target_val, pred))
                scores_rmse.append(root_mean_squared_error(target_val, pred))

        else:

            cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)  # TODO: n_splits in config
            scores_rmse = cross_val_score(estimator, X_data, target_data, scoring='neg_root_mean_squared_error', cv=cv,
                                          n_jobs=1)
            scores_mae = cross_val_score(estimator, X_data, target_data, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)

        if save_fit:
            directory = f'{self.fit_path}{"_".join([str(param) for param in self.descriptor_params])}'
            os.makedirs(directory, exist_ok=True)

            filename = f'GPR_z{kernel_degree}_opt_a{noise}_{self.descriptor_type}.sav'

            pickle.dump(estimator, open(os.path.join(directory, filename), 'wb'))
            print(f'Fit saved to {directory}.')
            print(f'Fit path: {self.fit_path}')

        metrics = {
            'Training MAE': np.mean(np.abs(scores_mae)),
            'MAE St. Dev.': np.std(np.abs(scores_mae)),
            'Training RMSE': np.mean(np.abs(scores_rmse)),
            'RMSE St. Dev.': np.std(np.abs(scores_rmse))
        }

        if report == 'full':

            print('Generating learning curves. May take some time.')
            print("_" * 65)

            self._plot_learning_curve(estimator, X_data, target_data, title=f'{self.descriptor_type}')

            print("{:<20} {:<10}".format("Metric", "Value (ppm)"))
            print("-" * 35)
            for key, value in metrics.items():
                print("{:<20} {:.0f}".format(key, value))

        elif report == 'errors':
            print("{:<20} {:<10}".format("Metric", "Value (ppm)"))
            print("-" * 35)
            for key, value in metrics.items():
                print("{:<20} {:.0f}".format(key, value))

        elif report is None:
            pass

        else:
            raise Exception(f"Unsupported option '{report}' for report. "
                            f"Specify as 'full' for printing error table "
                            f"and displaying learning curves, 'errors' for only"
                            f"printing error table leave at default for no report.")


        return np.mean(np.abs(scores_mae)), np.std(np.abs(scores_mae)), np.mean(np.abs(scores_rmse)), np.std(
            np.abs(scores_rmse)), opt_noise

    def gpr_test(self, kernel_degree, noise, report=None):

        folder = f'{self.fit_path}{"_".join([str(param) for param in self.descriptor_params])}'

        if os.path.isdir(folder):

            filename = f'GPR_z{kernel_degree}_opt_a{noise}_{self.descriptor_type}.sav'

            try:

                estimator = pickle.load(open(os.path.join(folder, filename), 'rb'))

            except Exception:

                raise Exception(f"Model with hyperparameters zeta={kernel_degree} "
                                f"and noise={noise} has not been trained yet. \n"
                                "Train specified model and save resulting fit before applying it on holdout set.")

        else:
            raise Exception(f"Model with descriptor parameters {self.descriptor_params} has not been trained yet. "
                            "Train specified model and save resulting fit before applying it on holdout set.")

        X_holdout = self.data_loader.load_samples()[1]

        target_holdout = self.data_loader.load_targets()[1]
        holdout_names = self.data_loader.load_targets()[3]

        predictions, std = estimator.predict(X_holdout, return_std=True)

        test_mae = mean_absolute_error(target_holdout, predictions)
        test_rmse = root_mean_squared_error(target_holdout, predictions)

        metrics = {'Test MAE': test_mae,
                   'Test RMSE': test_rmse}

        residuals = [observed - pred for pred, observed in zip(predictions, target_holdout)]

        if report == 'full':

            correlation = self._plot_correlation(predictions, target_holdout, threshold=test_rmse,
                                   title=f'{self.descriptor_type}', holdout_names=holdout_names)

            print('-'*35)
            print(f'Correlation: {correlation}')
            print(f'Uncertainties: {std}')
            print(f'Average Uncertainties: {np.mean(std)}')
            print('-' * 35)

            self._empirical_coverage(predictions, std, target_holdout)

            metrics = {'Test MAE': test_mae,
                       'Test RMSE': test_rmse,
                       '$R^{2}$': correlation}

            print('\n')
            print("{:<20} {:<10}".format("Metric", "Value (ppm)"))
            print("-" * 35)
            for key, value in metrics.items():
                print("{:<20} {:.0f}".format(key, value))

        elif report == 'errors':
            print('\n')
            print("{:<20} {:<10}".format("Metric", "Value (ppm)"))
            print("-" * 35)
            for key, value in metrics.items():
                print("{:<20} {:.0f}".format(key, value))

        return test_mae, test_rmse, predictions, std, residuals, holdout_names

    @staticmethod
    def _empirical_coverage(predictions, st_devs, target_holdout, z_score=1.96):
        CI_lower = [pred - (z_score * st_dev) for pred, st_dev in zip(predictions, st_devs)]
        CI_upper = [pred + (z_score * st_dev) for pred, st_dev in zip(predictions, st_devs)]

        n_vals = 0

        for target, lower_bound, upper_bound in zip(list(target_holdout), CI_lower, CI_upper):

            if lower_bound <= target <= upper_bound:
                n_vals += 1
            else:
                pass

        coverage = n_vals / len(target_holdout)

        print(f'Empirical coverage of model for holdout set: {coverage:.4f}')

        return coverage

    @staticmethod
    def _plot_learning_curve(estimator, X_data, target_data, title):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_data, target_data,
                                                                train_sizes=np.linspace(0.2, 1.0, 5),
                                                                cv=ShuffleSplit(n_splits=4, test_size=0.2,
                                                                random_state=42), scoring='neg_mean_absolute_error')

        train_test_diff_list = []

        for train_score, test_score in zip(train_scores, test_scores):
            diff = np.abs(np.mean(train_score) - np.mean(test_score))
            train_test_diff_list.append(diff)

        plt.figure()
        plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, -test_scores.mean(axis=1), 's-', color='g', label='Validation score')
        plt.fill_between(x=train_sizes, y1=-train_scores.mean(axis=1) - train_scores.std(axis=1),
                         y2=-train_scores.mean(axis=1) + train_scores.std(axis=1), color='r', alpha=0.3)
        plt.fill_between(x=train_sizes, y1=-test_scores.mean(axis=1) - test_scores.std(axis=1),
                         y2=-test_scores.mean(axis=1) + test_scores.std(axis=1), color='g', alpha=0.3)
        plt.xlabel('Number of training points', fontsize=16)
        plt.ylabel('MAE (ppm)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc='best', fontsize=14)
        plt.title(title, fontsize=18)
        plt.grid()
        plt.savefig(f'/home/alex/Pt_NMR/paper/figs/lc_{title}.png', dpi=400, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _plot_correlation(predictions, target_holdout, threshold, title, holdout_names, show_outliers=True):

        correlation = r2_score(target_holdout, predictions)

        residuals = [observed-pred for pred, observed in zip(predictions, target_holdout)]

        outliers = [(observed, pred, res, name) for observed, pred, res, name in zip(target_holdout, predictions, residuals, holdout_names) if
                    abs(res) > threshold]

        plt.scatter(target_holdout, predictions, edgecolors=(0, 0, 0))
        plt.plot([target_holdout.min(), target_holdout.max()], [target_holdout.min(), target_holdout.max()], 'k-',
                 lw=2)

        for observed, pred, res, name in outliers:
            plt.scatter(observed, pred, color='red')


        if show_outliers:

            residual_list = []

            print(f"Outliers ({len(outliers)}):\n------------")
            for observed, pred, res, outlier_name in outliers:
                print(f"Compound Name: {outlier_name}, Observed: {observed}, Predicted: {pred}, Residual: {res}")

                residual_list.append(abs(res))

            print(f"Maximum outlier deviation: {max(residual_list)}")
            print(f"Mean outlier deviation: {np.mean(residual_list)}")
            print(f"Standard deviation: {np.std(residual_list)}")

            print('Residuals:')
            print(residuals)

        plt.xlabel('Measured (ppm)', fontsize=16)
        plt.ylabel('Predicted (ppm)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f'{title} ($R^2$ = {correlation:.2f})', fontsize=18)
        plt.savefig(f'/home/alex/Pt_NMR/paper/figs/pp_{title}.png', dpi=400, bbox_inches='tight')
        plt.show()

        return correlation

    @staticmethod
    def _plot_hyperparameters(estimator):
        sigma_0_range = np.logspace(-5, 5, num=10)
        noise_level_range = np.logspace(-5, 5, num=10)
        sigma_0_grid, noise_level_grid = np.meshgrid(sigma_0_range, noise_level_range)
        log_marginal_likelihood = [estimator.log_marginal_likelihood(theta=np.log([0.36, sigma_value, noise]))
                                   for sigma_value, noise in zip(sigma_0_grid.ravel(), noise_level_grid.ravel())]
        log_marginal_likelihood = np.reshape(log_marginal_likelihood, newshape=noise_level_grid.shape)
        vmin, vmax = (-log_marginal_likelihood).min(), 10
        level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=10), decimals=1)
        plt.contour(sigma_0_grid, noise_level_grid, -log_marginal_likelihood, levels=level,
                    norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Sigma 0")
        plt.ylabel("Noise-level")
        plt.title("Log-marginal-likelihood")
        plt.show()

# TODO: add feature to generate SOAPs from SMILES strings
# TODO: Don't forget README before re-submitting
