import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation, RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from generate_descriptors import GenDescriptors


class SklearnGPRegressor(GenDescriptors):
    def __init__(self, descriptor_params, descriptor_path, central_atom, xyz_path, xyz_base,
                 descriptor_type, mode, target_path):
        """
       Initialize class for using descriptors as input for Gaussian Process Regression
       with cross-validated errors, hyperparameter tuning and visualization of learning curves

       :param mode: 'read' for passing pre-generated descriptors as input
       :param descriptor_type: Options currently implemented: 'SOAP' or 'APE-RF'
       :param descriptor_path: path where the pre-generated descriptors are stored
       :param descriptor_params: Parameters of the descriptors (SOAP: [rcut, nmax, lmax], APE-RF: [rcut, dim])
       :param central_atom: Atom symbol (str) of central atom ('Pt' for 195Pt-NMR)
       :param xyz_path: Path to directory where xyz-files are stored
       :param xyz_base: basename of the xyz_files (e.g. for st_1.xyz: 'st_')
       :param target_path: Path to target data (csv-file containing shift values)
       """

        super().__init__(descriptor_params, descriptor_path, central_atom, xyz_path, xyz_base, normalize=True)
        self.descriptor_type = descriptor_type
        self.mode = mode

        if isinstance(target_path, str):
            self.target_path = [target_path]
        elif isinstance(target_path, list):
            self.target_path = target_path
        else:
            raise ValueError("Paths should be a string or a list of strings")


    def gpr_train(self, kernel_degree, noise, lc=None, noise_estim=False, save_fit=True, stratify_train=True, ard=False):

        # TODO: Refactor such that methods for loading data are in separate module

        """
        Uses the sklearn implementation of Gaussian Process Regression. Defines GPR model with linear/
        polynomial kernel and evaluates a given hyperparameter combination (of the representation and the GPR model)
        on a given dataset using k-fold cross-validation. Provides learning curves for the training and validation set
        and option of optimizing the noise level based on the gradient of the log marginal likelihood (LML) (sklearn backend)

        :param kernel_degree: Degree of the polynomial kernel
        :param noise: Likelihood variance a. k. a. noise level of the data (is only added to K(X,X) of training points,
        noise is added to K(X,X) of test points when using WhiteKernel(), as done if noise_estim=True
        :param lc: Whether to generate and plot learning curve
        :param noise_estim: Whether to optimize the noise level using the LML. Needed for uncertainty estimates.
        :param save_fit: Whether to save the state of the fitted model

        :return: CV MAE and RMSE and corresponding standard deviations
        """

        X_data = self.load_samples()[0]
        target_data = self.load_targets(target_name='Experimental')[0]


        randomSeed = 42

        if kernel_degree == 1:

            if noise_estim:
                estimator = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(noise_level=noise),
                                                     random_state=randomSeed,
                                                     alpha=0.0, n_restarts_optimizer=10, normalize_y=True)

            else:
                estimator = GaussianProcessRegressor(kernel=DotProduct(), random_state=randomSeed, alpha=float(noise),
                                                     optimizer=None)

        elif kernel_degree > 1:

            if noise_estim:
                estimator = GaussianProcessRegressor(
                    kernel=Exponentiation(DotProduct(), int(kernel_degree)) + WhiteKernel(noise_level=noise),
                    random_state=randomSeed,
                    alpha=0.0, n_restarts_optimizer=10, normalize_y=True)

            else:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(sigma_0=0.0), int(kernel_degree)),
                                                     random_state=randomSeed, alpha=float(noise),
                                                     optimizer=None)

        else:

            estimator = GaussianProcessRegressor(kernel=ConstantKernel() * RBF() + WhiteKernel(noise_level=noise), alpha=0.0,
                                                 random_state=randomSeed, n_restarts_optimizer=10, normalize_y=True)

            if ard:

                X_data = StandardScaler().fit_transform(X_data)

                ard_kernel = ConstantKernel() * RBF(length_scale=np.ones(np.array(X_data).shape[1])) + WhiteKernel(noise_level=noise)

                estimator = GaussianProcessRegressor(kernel=ard_kernel, alpha=0.0,
                                                     random_state=randomSeed, n_restarts_optimizer=10, normalize_y=True)



        estimator.fit(X_data, target_data)

        if noise_estim:
            opt_noise = estimator.kernel_.k2.noise_level
            print(f'\n Optimized noise (on whole train set): {opt_noise}')

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

        else:
            opt_noise = None


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

            cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)
            scores_rmse = cross_val_score(estimator, X_data, target_data, scoring='neg_root_mean_squared_error', cv=cv,
                                          n_jobs=1)
            scores_mae = cross_val_score(estimator, X_data, target_data, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)

        if lc:
            self._plot_learning_curve(estimator, X_data, target_data, title='ChEAP')

        if save_fit:
            directory = f'/home/alex/Pt_NMR/data/fits/{"_".join([str(param) for param in self.descriptor_params])}'
            os.makedirs(directory, exist_ok=True)

            if noise_estim:
                filename = f'GPR_z{kernel_degree}_opt_a{noise}_{self.descriptor_type}.sav'

            else:
                filename = f'GPR_z{kernel_degree}_a{noise}_{self.descriptor_type}.sav'

            pickle.dump(estimator, open(os.path.join(directory, filename), 'wb'))

        return np.mean(np.abs(scores_mae)), np.std(np.abs(scores_mae)), np.mean(np.abs(scores_rmse)), np.std(
            np.abs(scores_rmse)), opt_noise

    def gpr_test(self, kernel_degree, noise, noise_estim=False, parity_plot=False, ecp=False):
        # TODO: Generalize saving of regressor
        folder = f'/home/alex/Pt_NMR/data/fits/{"_".join([str(param) for param in self.descriptor_params])}'

        if noise_estim:
            filename = f'GPR_z{kernel_degree}_opt_a{noise}_{self.descriptor_type}.sav'

        else:
            filename = f'GPR_z{kernel_degree}_a{noise}_{self.descriptor_type}.sav'

        estimator = pickle.load(open(os.path.join(folder, filename), 'rb'))

        X_holdout = DataLoader().load_samples()[1]

        target_holdout = DataLoader().load_targets()[1]
        holdout_names = DataLoader().load_targets()[3]

        predictions, std = estimator.predict(X_holdout, return_std=True)

        test_mae = mean_absolute_error(target_holdout, predictions)
        test_rmse = root_mean_squared_error(target_holdout, predictions)

        if parity_plot:
            self._plot_correlation(predictions, target_holdout, threshold=test_rmse,
                                   title='ChEAP', holdout_names=holdout_names,
                                   st_devs=std, show_outliers=True)

        if ecp:
            self._empirical_coverage(predictions, std, target_holdout)

        print('Errors on holdout test set (Backend: sklearn): \n-----------------------------------------')
        print(f'MAE: {test_mae} [ppm]')
        print(f'RMSE: {test_rmse} [ppm]')
        print('-----------------------------------------')

        return test_mae, test_rmse, predictions, std

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

        print(train_sizes, -train_scores.mean(axis=1), -test_scores.mean(axis=1))

        for train_score, test_score in zip(train_scores, test_scores):
            diff = np.abs(np.mean(train_score) - np.mean(test_score))
            train_test_diff_list.append(diff)

        print('Train/Test Score differences: \n -----------------------------------------')
        print(train_test_diff_list)

        plt.figure()
        plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, -test_scores.mean(axis=1), 's-', color='g', label='Test score')
        plt.fill_between(x=train_sizes, y1=-train_scores.mean(axis=1) - train_scores.std(axis=1),
                         y2=-train_scores.mean(axis=1) + train_scores.std(axis=1), color='r', alpha=0.3)
        plt.fill_between(x=train_sizes, y1=-test_scores.mean(axis=1) - test_scores.std(axis=1),
                         y2=-test_scores.mean(axis=1) + test_scores.std(axis=1), color='g', alpha=0.3)
        plt.xlabel('Number of training points', fontsize=12)
        plt.ylabel('MAE (ppm)', fontsize=12)
        plt.legend(loc='best')
        plt.title(title, fontsize=18)
        plt.grid()
        plt.savefig(f'/home/alex/Desktop/lc_{title}.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _plot_correlation(predictions, target_holdout, threshold, title, holdout_names, st_devs, show_outliers=False):

        correlation = r2_score(target_holdout, predictions)

        residuals = [abs(pred) - abs(observed) for pred, observed in zip(predictions, target_holdout)]

        outliers = [(observed, pred, res, name) for observed, pred, res, name in zip(target_holdout, predictions, residuals, holdout_names) if
                    abs(res) > threshold]

        plt.scatter(target_holdout, predictions, edgecolors=(0, 0, 0))
        plt.plot([target_holdout.min(), target_holdout.max()], [target_holdout.min(), target_holdout.max()], 'k-',
                 lw=2)

        ok_num = 0
        for res, std in zip(residuals, st_devs):
            if abs(res) < abs(std):
                symb = 'O.K.'
                ok_num += 1
            else:
                symb = 'PROBLEM'

            print(res, ' <---->', std, symb)

        print(ok_num / len(residuals))

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

        plt.xlabel('Measured (ppm)', fontsize=12)
        plt.ylabel('Predicted (ppm)', fontsize=12)
        plt.title(f'{title} ($R^2$ = {correlation:.2f})', fontsize=16)
        plt.savefig(f'/home/alex/Desktop/pp_{title}.png', dpi=300, bbox_inches='tight')
        plt.show()

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
