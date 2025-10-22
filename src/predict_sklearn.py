import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation, RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from base import BaseConfig
from omegaconf import OmegaConf
from data_loader import DataLoader


class SklearnGPRegressor(BaseConfig):
    """Gaussian Process Regressor for molecular property prediction.

        This class implements GP regression with scikit-learn for predicting NMR
        chemical shifts from molecular descriptors. Provides training with cross-validation,
        testing with uncertainty quantification, and visualization of results.

        Parameters
        ----------
        config : omegaconf.DictConfig
            Hydra configuration object containing:
            - config.mode : str
                'read' or 'write' for descriptor loading/generation
            - config.backend.model.target_path : str or list
                Path(s) to target value CSV files
            - config.backend.model.fit_path : str
                Path for saving trained models
            - All DataLoader configuration parameters

        Attributes
        ----------
        data_loader : DataLoader
            Instance for loading descriptors and targets.
        mode : str
            Operating mode ('read' or 'write').
        target_path : list of str
            Paths to target CSV files.
        fit_path : str
            Directory for saving trained models.

        Notes
        -----
        Models are saved as pickle files with suffix '_fit.pkl'.
        Supports multiple kernel types and hyperparameter optimization.
        """

    def __init__(self, config):
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

    def gpr_train(self, kernel_degree, noise, save_fit=True, stratify_train=True, report=None):

        """ Uses the sklearn implementation of Gaussian Process Regression. Defines GPR model with linear/
        polynomial kernel and evaluates a given hyperparameter combination (of the representation and the GPR model)
        on a given dataset using k-fold cross-validation. Provides learning curves for the training and validation set
        and option of optimizing the noise level based on the gradient of the log marginal likelihood (LML) (sklearn backend)

        Parameters
        ----------
        kernel_degree : int
            Degree of the polynomial kernel
        noise : float
            Likelihood variance a. k. a. noise variance of the labels (is only added to K(X,X) of training points,
            noise is added to K(X,X) of test points when using WhiteKernel()
        save_fit : bool
            Whether to save the state of the fitted model
        stratify_train : bool
            Whether the train/validation splits in k-fold CV are stratified or not.
        report : str
            Which performance metrics to output. "Errors" for mean errors only and "full" for mean errors and learning curve

        Returns
        -------
        train_mae : float
            Mean cross-validated MAE on the training set.
        train_mae_std : float
            Standard deviation of the cross-validated MAE on the training set.
        train_rmse : float
            Mean cross-validated RMSE on the training set.
        train_rmse_std : float
            Standard deviation of the cross-validated RMSE on the training set.
        opt_noise : float
            Optimized noise variance on the whole training set

        Raises
        ------
        np.linalg.LinAlgError
            If kernel matrix is singular.
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
                alpha=0.0, n_restarts_optimizer=10, normalize_y=True)  # always normalize labels

        else:

            estimator = GaussianProcessRegressor(kernel=ConstantKernel() * RBF() + WhiteKernel(noise_level=noise), alpha=0.0,
                                                 random_state=randomSeed, n_restarts_optimizer=10, normalize_y=True)

        print('Training in progress....')
        print(f'Init noise: {noise}')

        estimator.fit(X_data, target_data)

        opt_noise = estimator.kernel_.k2.noise_level
        print(f'\nOptimized noise variance: {opt_noise:.3f} \n')

        if kernel_degree == 0:
            opt_const = 0
            opt_lengthscale = estimator.kernel_.k1.k2.length_scale
            print(f'Optimized RBF lengthscale: {opt_lengthscale:.3f} \n')

        elif kernel_degree == 1:
            opt_const = estimator.kernel_.k1.sigma_0
        else:
            opt_const = estimator.kernel_.k1.kernel.sigma_0

        print(f'Optimized kernel bias: {opt_const:.3f} \n')

        lml = estimator.log_marginal_likelihood_value_
        print(f'Log marginal likelihood: {lml:.3f} \n')

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

        if save_fit:
            directory = f'{self.fit_path}{self.descriptor_type}_{"_".join([f'{self.descriptor_params[param]}' for param in self.descriptor_params])}'
            os.makedirs(directory, exist_ok=True)

            filename = f'GPR_z{kernel_degree}_opt_a{noise}.sav'

            with open(os.path.join(directory, filename), 'wb') as f:
                pickle.dump(estimator, f)

            print(f'Fit saved to {directory}. \n')

        metrics = {
            'Training MAE': np.mean(np.abs(scores_mae)),
            'MAE St. Dev.': np.std(np.abs(scores_mae)),
            'Training RMSE': np.mean(np.abs(scores_rmse)),
            'RMSE St. Dev.': np.std(np.abs(scores_rmse))
        }

        if report == 'full':

            print('Generating learning curves. May take some time. \n')
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
                print("{:<20} {:.0f}".format(key, value), "\n")

        elif report is None:
            pass

        else:
            raise Exception(f"Unsupported option '{report}' for report. "
                            f"Specify as 'full' for printing error table "
                            f"and displaying learning curves, 'errors' for only"
                            f"printing error table leave at default for no report. \n")

        train_mae = np.mean(np.abs(scores_mae))
        train_mae_std = np.std(np.abs(scores_mae))
        train_rmse = np.mean(np.abs(scores_rmse))
        train_rmse_std = np.std(np.abs(scores_rmse))

        return train_mae, train_mae_std, train_rmse, train_rmse_std, opt_noise

    def gpr_test(self, kernel_degree, noise, report=None):
        """Test trained GP model on holdout set.

        Loads a trained GP model and evaluates it on the holdout test set. Computes
        predictions with uncertainty estimates and optionally generates
        parity plots and coverage probability.

        Parameters
        ----------
        kernel_degree : int
            Degree of the polynomial kernel.
        noise : float
            Likelihood variance a. k. a. noise variance of the labels (is only added to K(X,X) of training points,
            noise is added to K(X,X) of test points when using WhiteKernel()
        report : str
            Which performance metrics to output. "Errors" for mean errors only and "full" for mean errors, parity plots
            and coverage probability.

        Returns
        -------
        mae_test : float
            Mean absolute error on test set.
        rmse_test : float
            Root mean squared error on test set.
        std_test : float
            Mean prediction uncertainty (standard deviation).

        Notes
        -----
        Requires a trained model saved by gpr_train.
        """

        folder = f'{self.fit_path}{self.descriptor_type}_{"_".join([f'{self.descriptor_params[param]}' for param in self.descriptor_params])}'

        if os.path.isdir(folder):

            filename = f'GPR_z{kernel_degree}_opt_a{noise}.sav'

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

        residuals = [observed-pred for pred, observed in zip(predictions, target_holdout)]

        if report == 'full':

            correlation = self._plot_correlation(predictions, target_holdout, threshold=test_rmse,
                                   title=f'{self.descriptor_type}', holdout_names=holdout_names)

            print('-'*35)
            print(f'Correlation: {correlation:.2f}')
            print(f'Average Uncertainty (ppm): {np.mean(std):.0f}')
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
        """Calculate empirical coverage of prediction intervals.

       Computes the fraction of true values within confidence intervals
       defined by predicted mean ± z * predicted std.

       Parameters
       ----------

       predictions : list
            List of predictions on the holdout set.
       st_devs : list
            List of predictive standard deviations (uncertainties) on the holdout set.
       target_holdout : list
            List of true label values of the holdout set.
       z_score : float
            Value of the z-score used to define the confidence interval.
            Corresponds to nominal coverage probability (1.96 for 95%)

       Returns
       -------
       coverage : float
            Empirical coverage probability of the model on the holdout set.
       """

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
        """Generate learning curves for a given model

        Plots the Monte-Carlo cross-validated (ShuffleSplit()) training and validation MAE
        for different training set sizes. For each training set size, the validation set size
        amounts to 20% of the training set size.

        Parameters
        ----------

        estimator : GaussianProcessRegressor
            The regressor with hyperparameters as specified in gpr_train.

        X_data : array-like
            Array of shape (n_samples, n_features) containing the training inputs.

        target_data : array-like
            Array of shape (n_samples, 1) containing the training labels (targets).

        title : str
            Title of the plot and filename of the saved figure. Descriptor type by default.

        Returns
        -------
        None
        """
        warnings.filterwarnings('ignore')

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
        """Generate parity plot for a given model

        Plots the observed vs. predicted values of the holdout set and provides the correlation coefficient.
        Predictions that deviate from the observed values more than the holdout RMSE are flagged as outliers
        and plotted in red.

        Parameters
        ----------

        predictions : list
            List of a given model's predictions on the holdout set.

        target_holdout : list
            List of true label values of the holdout set.

        threshold : float
            Threshold value for outlier detection (Holdout RMSE by default).

        title : str
            Title of the plot and filename of the saved figure. Descriptor type by default.

        holdout_names : list
            List of compound names corresponding to the holdout set.

        show_outliers : bool
            Whether to flag outliers in the plot

        Returns
        -------

        correlation : float
            Correlation coefficient squared between observed and predicted values.
        """

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

            print(f"Maximum outlier deviation: {max(residual_list):.0f}")
            print(f"Mean outlier deviation: {np.mean(residual_list):.0f}")
            print(f"Standard deviation: {np.std(residual_list):.0f}")

        plt.xlabel('Measured (ppm)', fontsize=16)
        plt.ylabel('Predicted (ppm)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f'{title} ($R^2$ = {correlation:.2f})', fontsize=18)
        plt.savefig(f'/home/alex/Pt_NMR/paper/figs/pp_{title}.png', dpi=400, bbox_inches='tight')
        plt.show()

        return correlation
