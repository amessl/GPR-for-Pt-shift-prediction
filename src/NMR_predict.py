import numpy as np
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation, RBF, WhiteKernel
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import Normalizer
import gpytorch
import torch
from gpytorch_nmr import gpr_estimator
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from generate_descriptors import generate_descriptors
import pickle



class GPR_NMR(generate_descriptors):
    def __init__(self, descriptor_params, descriptor_path, central_atom, xyz_path, xyz_base,
                 descriptor_type, mode, target_path):

        """
       Initialize class for using descriptors as input for kernel regression (GPR and KRR)
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

    def read_descriptors(self, path_index):

        """
        Read descriptors that were already generated from corresponding folder as specified
        when creating an instance of this class.

        :param path_index: Integer number for iterating over list of paths
        when partitioning data into train and test set (they are stored in individual folders)

        :return:
        n x p-array of feature vectors, where n is the number of samples (structures) and p the
        dimensionality of each feature vector ("Design matrix")
        """

        dataset = []

        descriptor_path = os.path.join(self.descriptor_path[path_index],
                                       '_'.join(str(param) for param in self.descriptor_params))
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

        if self.normalize:
            dataset = Normalizer(norm='l2').fit_transform(dataset)

        # print(
        #    f'Descriptor files read: {len(descriptor_filenames)} \nAverage size: {round((memory / file_count) / 1024, 3)} kB \n')
        print(f'Dimensions of design matrix: {np.shape(dataset)}')

        return dataset

    def load_samples(self, partitioned=True):
        """
        Loads samples (representations a. k. a. feature vectors) from pre-generated files
        or generating them, depending on the "mode" attribute ("read" or "write")

        :param partitioned: Whether to load train and test samples separately (Default=True)
        :return: Design matrix of total dataset or training and test samples (holdout)
        """

        X_holdout = None

        if self.mode == 'read':

            X_data = []

            for path_index in range(0, len(self.descriptor_path)):
                X_data.append(self.read_descriptors(path_index))

            if partitioned:
                X_data = X_data[0]
                X_holdout = X_data[1]

            else:
                X_data = X_data  # total dataset

        elif self.mode == 'write':

            if self.descriptor_type == 'SOAP':

                if partitioned:
                    X_data = self.generate_SOAPs_partitioned()[0]
                    X_holdout = self.generate_SOAPs_partitioned()[1]
                else:
                    X_data = self.generate_SOAPs()

            elif self.descriptor_type == 'APE-RF':

                if partitioned:
                    X_data = self.get_APE_RF_partitioned()[0]
                    X_holdout = self.get_APE_RF_partitioned()[1]
                else:
                    X_data = self.get_APE_RF()

            elif self.descriptor_type == 'SIF':

                if partitioned:
                    X_data = self.get_SIF_partitioned()[0]
                    X_holdout = self.get_SIF_partitioned()[1]

                else:
                    X_data = self.get_SIF()

            else:
                raise Exception('Descriptor type has to be specified. Use "SOAP" or "APE-RF"')

        else:
            raise Exception('Mode has to be specified as "read" for using pre-generated descriptors \n'
                            'or "write" for generating new descriptors and passing them as input"')

        return X_data, X_holdout

    def load_targets(self, target_name='Experimental', partitioned=True):

        """
        Loads target values (chemical shifts) from csv-file
        :param target_name: Name of the column in the corresponding csv-file (Default='Experimental')
        :param partitioned: Whether to load train and test samples separately (Default=True)
        :return: Total number of targets or training and test targets
        """

        if partitioned:

            target_training_data = pd.read_csv(f'{self.target_path[0]}')
            target_test_data = pd.read_csv(f'{self.target_path[1]}')

            sorted_train = target_training_data.sort_values(by='Index')
            sorted_test = target_test_data.sort_values(by='Index')

            target_data = sorted_train[str(target_name)]
            target_holdout = sorted_test[str(target_name)]


        else:
            indexed_target_data = pd.read_csv(f'{self.target_path[0]}')
            sorted_train = indexed_target_data.sort_values(by='Index')

            target_data = sorted_train[str(target_name)]
            target_holdout = None

        return target_data, target_holdout, sorted_train['Index']

    def GPR_train(self, kernel_degree, noise, lc=None, noise_estim=False, save_fit=True):

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
                                                     alpha=0.0, n_restarts_optimizer=5, normalize_y=True)

            else:
                estimator = GaussianProcessRegressor(kernel=DotProduct(), random_state=randomSeed, alpha=float(noise),
                                                     optimizer=None)

        elif kernel_degree > 1:

            if noise_estim:
                estimator = GaussianProcessRegressor(
                    kernel=Exponentiation(DotProduct(), int(kernel_degree)) + WhiteKernel(noise_level=noise),
                    random_state=randomSeed,
                    alpha=0.0, n_restarts_optimizer=5, normalize_y=True)

            else:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(), int(kernel_degree)),
                                                     random_state=randomSeed, alpha=float(noise),
                                                     optimizer=None)

        else:

            estimator = GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=noise), alpha=0.0,
                                                 random_state=randomSeed, n_restarts_optimizer=1, normalize_y=False)

        estimator.fit(X_data, target_data)

        if noise_estim:
            opt_noise = estimator.kernel_.k2.noise_level
            print(f'\n Optimized noise: {opt_noise} \n')



        else:
            opt_noise = None

        cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)

        scores_rmse = cross_val_score(estimator, X_data, target_data, scoring='neg_root_mean_squared_error', cv=cv,
                                      n_jobs=1)
        scores_mae = cross_val_score(estimator, X_data, target_data, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)

        if lc:
            self._plot_learning_curve(estimator, X_data, target_data, title=self.descriptor_type)

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



    def GPR_train_gpytorch(self, kernel_degree, noise, n_splits, noise_estim=False):

        # TODO: learning curve and saving of regressor

        X_data = self.load_samples()[0]
        target_data = self.load_targets(target_name='Experimental')[0]

        target_indices = self.load_targets()[2]

        target_data = np.array(sorted(list(target_data), key=lambda x: list(target_indices)))

        # Define the kernel
        if kernel_degree == 1:
            kernel = gpytorch.kernels.LinearKernel()

        elif kernel_degree > 1:
            kernel = gpytorch.kernels.PolynomialKernel(power=kernel_degree)
        else:
            kernel = gpytorch.kernels.RBFKernel()

        scaled_kernel = gpytorch.kernels.ScaleKernel(kernel.double())

        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        likelihood.noise = torch.tensor(noise)

        randomSeed = 42
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=randomSeed)

        MAE_list = []
        RMSE_list = []


        for fold, (train_idx, val_idx) in enumerate(kfold.split(X=X_data, y=target_data)):

            # Code adapted based on git christianversloot/machine-learning-articles

            # Split data
            train_x, train_y = X_data[train_idx], target_data[train_idx]
            val_x, val_y = X_data[val_idx], target_data[val_idx]

            # Convert to PyTorch tensors (use consistent dtype)
            train_x = torch.tensor(train_x, dtype=torch.float64)
            train_y = torch.tensor(train_y, dtype=torch.float64)
            val_x = torch.tensor(val_x, dtype=torch.float64)
            val_y = torch.tensor(val_y, dtype=torch.float64)

            # Initialize the model
            model = gpr_estimator(train_x, train_y, likelihood, scaled_kernel)


            if noise_estim:
                model.train()
                likelihood.train()

                print('Optimizing noise using MLL')
                optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)

                # Loss for GPs - Marginal Log Likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                prev_loss = float('inf')
                tol = 1e-3  # Tolerance for convergence

                training_iterations = 10000
                for i in range(training_iterations):
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()

                    # Print the loss and current noise level
                    noise = likelihood.noise.item()
                    print(noise)
                    # Check convergence
                    if abs(prev_loss - loss.item()) < tol:
                        print(f"Convergence achieved!, Noise: {noise}")
                        break

                    prev_loss = loss.item()

                model.eval()
                likelihood.eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = likelihood(model(val_x))
                    MAE = torch.mean(torch.abs(predictions.mean - val_y)).item()
                    RMSE = torch.sqrt(torch.mean(torch.abs(predictions.mean - val_y)**2)).item()
                    RMSE_list.append(RMSE)
                    MAE_list.append(MAE)

            else:

                model.eval()
                likelihood.eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = likelihood(model(val_x))
                    MAE = torch.mean(torch.abs(predictions.mean - val_y)).item()
                    RMSE = torch.sqrt(torch.mean(torch.abs(predictions.mean - val_y)**2)).item()
                    RMSE_list.append(RMSE)
                    MAE_list.append(MAE)

        return torch.tensor(MAE_list).mean().item(), torch.tensor(MAE_list).std().item(), torch.tensor(RMSE_list).mean().item(), torch.tensor(MAE_list).std().item()


    def GPR_test(self, kernel_degree, noise, noise_estim=False, parity_plot=False, ecp=False):

        # TODO: Generalize saving of regressor
        folder = f'/home/alex/Pt_NMR/data/fits/{"_".join([str(param) for param in self.descriptor_params])}'

        if noise_estim:
            filename = f'GPR_z{kernel_degree}_opt_a{noise}_{self.descriptor_type}.sav'

        else:
            filename = f'GPR_z{kernel_degree}_a{noise}_{self.descriptor_type}.sav'

        estimator = pickle.load(open(os.path.join(folder, filename), 'rb'))

        X_holdout = self.load_samples()[1]
        target_holdout = self.load_targets()[1]

        predictions, std = estimator.predict(X_holdout, return_std=True)

        test_mae = mean_absolute_error(target_holdout, predictions)
        test_rmse = root_mean_squared_error(target_holdout, predictions)

        if parity_plot:
            self._plot_correlation(predictions, target_holdout, threshold=test_rmse, show_outliers=True)

        if ecp:
            self._empirical_coverage(predictions, std, target_holdout)

        print('Errors on holdout test set (Backend: sklearn): \n-----------------------------------------')
        print(f'MAE: {test_mae} [ppm]')
        print(f'RMSE: {test_rmse} [ppm]')
        print('-----------------------------------------')

        return test_mae, test_rmse

    def GPR_test_gpytorch(self, kernel_degree, noise):

        X_data = self.load_samples()[0]
        target_data = self.load_targets(target_name='Experimental')[0]
        target_indices = self.load_targets()[2]

        X_data_train = torch.tensor(X_data)
        target_data_train = torch.tensor(np.array(sorted(list(target_data), key=lambda x: list(target_indices))))

        # Define the kernel
        if kernel_degree == 1:
            kernel = gpytorch.kernels.LinearKernel()

        elif kernel_degree > 1:
            kernel = gpytorch.kernels.PolynomialKernel(power=kernel_degree)
        else:
            kernel = gpytorch.kernels.RBFKernel()

        scaled_kernel = gpytorch.kernels.ScaleKernel(kernel.double())

        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        likelihood.noise = torch.tensor(noise)

        model = gpr_estimator(X_data_train, target_data_train, likelihood, kernel=scaled_kernel)

        X_data_test = torch.tensor(self.load_samples()[1])
        target_data_test = torch.tensor(np.array(self.load_targets()[1]))

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(X_data_test))
            test_MAE = torch.mean(torch.abs(predictions.mean - target_data_test)).item()
            test_RMSE = torch.sqrt(torch.mean(torch.abs(predictions.mean - target_data_test) ** 2)).item()


        print('Errors on holdout test set (Backend: GPyTorch): \n-----------------------------------------')
        print(f'MAE: {test_MAE} [ppm]')
        print(f'RMSE: {test_RMSE} [ppm]')
        print('-----------------------------------------')

        return test_MAE, test_RMSE


    def _empirical_coverage(self, predictions, st_devs, target_holdout, z_score=1.96):
        CI_lower = [pred - (z_score * st_dev) for pred, st_dev in zip(predictions, st_devs)]
        CI_upper = [pred + (z_score * st_dev) for pred, st_dev in zip(predictions, st_devs)]

        n_vals = 0

        for target, lower_bound, upper_bound in zip(list(target_holdout), CI_lower, CI_upper):

            if lower_bound <= target <= upper_bound:
                n_vals += 1
            else:
                pass

        coverage = n_vals / len(target_holdout)

        print(f'Empirical coverage of model for holdout set: {coverage}')

        return coverage

    def _plot_learning_curve(self, estimator, X_data, target_data, title):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_data, target_data,
                                                                train_sizes=np.linspace(0.2, 1.0, 5),
                                                                # cv=ShuffleSplit(n_splits=4, test_size=0.25,
                                                                cv=KFold(n_splits=4, shuffle=True,
                                                                         random_state=42),
                                                                scoring='neg_mean_absolute_error')

        # TODO: fix CV in learning curves

        plt.figure()
        plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, -test_scores.mean(axis=1), 'o-', color='g', label='Test score')
        plt.fill_between(x=train_sizes, y1=-train_scores.mean(axis=1) - train_scores.std(axis=1),
                         y2=-train_scores.mean(axis=1) + train_scores.std(axis=1), color='r', alpha=0.3)
        plt.fill_between(x=train_sizes, y1=-test_scores.mean(axis=1) - test_scores.std(axis=1),
                         y2=-test_scores.mean(axis=1) + test_scores.std(axis=1), color='g', alpha=0.3)
        plt.xlabel('Training examples')
        plt.ylabel('MAE [ppm]')
        plt.legend(loc='best')
        plt.title(title, fontsize=18)
        plt.grid()
        plt.savefig(f'/home/alex/Desktop/lc_{title}.png', dpi=500, bbox_inches='tight')
        plt.show()

    def _plot_correlation(self, predictions, target_holdout, threshold, show_outliers=False):

        correlation = r2_score(target_holdout, predictions)

        target_data = pd.read_csv(f'{self.target_path[1]}')

        residuals = [observed - pred for observed, pred in zip(target_holdout, predictions)]

        outliers = [(observed, pred, res) for observed, pred, res in zip(target_holdout, predictions, residuals) if
                    abs(res) > threshold]

        plt.scatter(target_holdout, predictions, edgecolors=(0, 0, 0))
        plt.plot([target_holdout.min(), target_holdout.max()], [target_holdout.min(), target_holdout.max()], 'k--',
                 lw=4)
        # TODO: Check reasonability of the line plot in _plot_correlation

        outlier_names = []

        shifts = target_data['Experimental']
        compound_names = target_data['Name']

        for observed, pred, res in outliers:
            for shift, compound_name in zip(shifts, compound_names):
                if observed == shift:
                    outlier_names.append(compound_name)

            plt.scatter(observed, pred, color='red')

        if show_outliers:

            print(f"Outliers ({len(outlier_names)}):\n------------")
            for (observed, pred, res), outlier_name in zip(outliers, outlier_names):
                print(f"Compound Name: {outlier_name}, Observed: {observed}, Predicted: {pred}, Residual: {res}")

        plt.xlabel('Measured [ppm]')
        plt.ylabel('Predicted [ppm]')
        plt.title(f'Parity plot ($R^2$ = {correlation:.2f})')
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
        plt.contour(sigma_0_grid, noise_level_grid, -log_marginal_likelihood, levels=level,
                    norm=LogNorm(vmin=vmin, vmax=vmax))
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
        grid_search = RandomizedSearchCV(estimator, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1,
                                         n_iter=100)
        grid_search.fit(train_X, train_target)
        best_params = grid_search.best_params_
        best_mae = -grid_search.best_score_
        print('Best params:', best_params)
        print('Best MAE:', best_mae)

# TODO: add feature to generate SOAPs from SMILES strings
# TODO: Don't forget README before re-submitting
