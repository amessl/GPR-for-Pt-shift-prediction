import numpy as np
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation, RBF, WhiteKernel
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
       """

        super().__init__(descriptor_params, descriptor_path, central_atom, xyz_path, xyz_base)
        self.mode = mode
        self.descriptor_type = descriptor_type

    def read_descriptors(self, path_index):

        """
        Read descriptors that were already generated from corresponding folder as specified
        when creating an instance of this class.

        :return:
        n x p-array of descriptor vectors, where n is the number of samples (structures) and p the
        dimensionality of each descriptor vector ("Design matrix")
        """

        dataset = []

        descriptor_path = os.path.join(self.descriptor_path[path_index], '_'.join(str(param) for param in self.descriptor_params))
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
            f'Descriptor files read: {len(descriptor_filenames)} \nAverage size: {round((memory / file_count) / 1024, 3)} kB \n')
        print(f'Dimensions of design matrix: {np.shape(dataset)}')

        return dataset


    def GPR_predict(self, kernel_degree, target_path, target_name, normalize, noise,
                partitioned=False, lc=None, parity_plot=None, noise_estim=False):

        if self.mode == 'read':

            X_data = []

            for path_index in range(0, len(self.descriptor_path)):
                X_data.append(self.read_descriptors(path_index))

            X_data = X_data[0]  # only training data

        elif self.mode == 'write':

            if self.descriptor_type == 'SOAP':

                if partitioned:
                    X_data = self.generate_SOAPs_partitioned()[0]
                else:
                    X_data = self.generate_SOAPs()

            elif self.descriptor_type == 'APE-RF':

                if partitioned:
                    X_data = self.get_APE_RF_partitioned()[0]
                else:
                    X_data = self.get_APE_RF(smooth_cutoff=False)

            elif self.descriptor_type == 'SIF':

                X_data = self.get_SIF()

            else:
                raise Exception('Descriptor type has to be specified. Use "SOAP" or "APE-RF"')

        else:
            raise Exception('Mode has to be specified as "read" for using pre-generated descriptors \n'
                            'or "write" for generating new descriptors and passing them as input"')

        if normalize:
            X_data = Normalizer(norm='l2').fit_transform(X_data)

        if partitioned:

            target_training_data = pd.read_csv(f'{target_path[0]}')
            target_test_data = pd.read_csv(f'{target_path[1]}')

            sorted_train = target_training_data.sort_values(by='Index')
            sorted_test = target_test_data.sort_values(by='Index')

            target_data = sorted_train[str(target_name)]

        else:
            indexed_target_data = pd.read_csv(f'{target_path[0]}')
            sorted_train = indexed_target_data.sort_values(by='Index')

            target_data = sorted_train[str(target_name)]

        randomSeed = 42

        if kernel_degree == 1:

            if noise_estim:
                estimator = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(noise_level=noise),
                                                     random_state=randomSeed,
                                                     alpha=0.0, n_restarts_optimizer=10)

                print(f'Optimized noise level: {estimator.kernel.k2.get_params(["noise_level"])}')

            else:
                estimator = GaussianProcessRegressor(kernel=DotProduct(), random_state=randomSeed, alpha=float(noise),
                                                     optimizer=None)

        elif kernel_degree > 1:

            if noise_estim:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(), int(kernel_degree)) + WhiteKernel(noise_level=noise),
                                                     random_state=randomSeed,
                                                     alpha=0.0, n_restarts_optimizer=10)

                print(f'Optimized noise level: {estimator.kernel.k2.get_params(["noise_level"])}')

            else:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(), int(kernel_degree)), random_state=randomSeed,alpha=float(noise),
                                                     optimizer=None)

        else:
            estimator = GaussianProcessRegressor(kernel=RBF(), alpha=float(noise))

        cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)
        #cv = ShuffleSplit(n_splits=4, random_state=randomSeed, test_size=0.25, )
        scores_rmse = cross_val_score(estimator, X_data, target_data, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=1)
        print('RMSE (4-fold CV):', np.mean(np.abs(scores_rmse)), '[ppm]', np.std(np.abs(scores_rmse)), '[ppm] (STDEV)')

        scores_mae = cross_val_score(estimator, X_data, target_data, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)
        print('MAE (4-fold CV):', np.mean(np.abs(scores_mae)), '[ppm]', np.std(np.abs(scores_mae)), '[ppm] (STDEV)')

        if lc:
            self._plot_learning_curve(estimator, X_data, target_data, title=self.descriptor_type)

        if parity_plot:
            #self._plot_correlation(target_path, estimator, train_X, test_X, train_target, test_target, threshold=np.mean(np.abs(scores_rmse)))
             print('Parity plot not supported atm.')
        return np.mean(np.abs(scores_mae)), np.std(np.abs(scores_mae)), np.mean(np.abs(scores_rmse)), np.std(np.abs(scores_rmse))


    def KRR_predict(self, kernel_degree, target_path, target_name, normalize, hold_out, alpha,
                partitioned=False, lc=None, parity_plot=None):

        if self.mode == 'read':

            X_data = []

            for path_index in range(0, len(self.descriptor_path)):
                X_data.append(self.read_descriptors(path_index))

            X_data = X_data[0]  # only training data

        elif self.mode == 'write':

            if self.descriptor_type == 'SOAP':

                X_data = self.generate_SOAPs()

            elif self.descriptor_type == 'APE-RF':

                X_data = self.get_APE_RF()

            else:
                raise Exception('Descriptor type has to be specified. Use "SOAP" or "APE-RF"')

        else:
            raise Exception('Mode has to be specified as "read" for using pre-generated descriptors \n'
                            'or "write" for generating new descriptors and passing them as input"')

        if normalize:
            X_data = Normalizer(norm='l2').fit_transform(X_data)

        target_data = pd.read_csv(f'{target_path}.csv')[str(target_name)]

        randomSeed = 42
        train_X, test_X, \
            train_target, test_target = train_test_split(X_data, target_data,
                                                         random_state=randomSeed, test_size=0.25, shuffle=True)


        if kernel_degree == 1:
            estimator = Ridge(random_state=randomSeed, alpha=float(alpha))

        elif kernel_degree > 1:
            estimator = KernelRidge(kernel='polynomial', degree=int(kernel_degree),
                                    alpha=float(alpha))

        else:
            estimator = KernelRidge(kernel='rbf', alpha=float(alpha))


        estimator.fit(train_X, train_target)

        #cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)
        cv = ShuffleSplit(n_splits=4, random_state=randomSeed, test_size=0.25)

        scores_rmse = cross_val_score(estimator, X_data, target_data, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=1)
        print('RMSE (4-fold CV):', np.mean(np.abs(scores_rmse)), '[ppm]', np.std(np.abs(scores_rmse)), '[ppm] (STDEV)')

        scores_mae = cross_val_score(estimator, X_data, target_data, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)
        print('MAE (4-fold CV):', np.mean(np.abs(scores_mae)), '[ppm]', np.std(np.abs(scores_mae)), '[ppm] (STDEV)')

        if lc:
            self._plot_learning_curve(estimator, X_data, target_data, title=self.descriptor_type)

        if parity_plot:
            self._plot_correlation(target_path, estimator, train_X, test_X, train_target, test_target, threshold=np.mean(np.abs(scores_rmse)))


        return np.mean(np.abs(scores_mae)), np.std(np.abs(scores_mae)), np.mean(np.abs(scores_rmse)), np.std(np.abs(scores_rmse))

    def _plot_learning_curve(self, estimator, X_data, target_data, title):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_data, target_data,
                                                                train_sizes=np.linspace(0.2, 1.0, 5),
                                                                #cv=ShuffleSplit(n_splits=4, test_size=0.25,
                                                                cv=KFold(n_splits=4, shuffle=True,
                                                                random_state=42), scoring='neg_mean_absolute_error')

        # TODO: fix CV in learning curves

        plt.figure()
        plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, -test_scores.mean(axis=1), 'o-', color='g', label='Test score')
        plt.fill_between(x=train_sizes, y1=-train_scores.mean(axis=1)-train_scores.std(axis=1),
                         y2=-train_scores.mean(axis=1)+train_scores.std(axis=1), color='r', alpha=0.3)
        plt.fill_between(x=train_sizes, y1=-test_scores.mean(axis=1)-test_scores.std(axis=1),
                         y2=-test_scores.mean(axis=1)+test_scores.std(axis=1), color='g', alpha=0.3)
        plt.xlabel('Training examples')
        plt.ylabel('MAE [ppm]')
        plt.legend(loc='best')
        plt.title(title, fontsize=18)
        plt.grid()
        plt.savefig(f'/home/alex/Desktop/lc_{title}.png', dpi=500, bbox_inches='tight')
        plt.show()

    def _plot_correlation(self, target_path, estimator, train_X, test_X,
                          train_target, test_target, threshold):
        estimator.fit(train_X, train_target)
        prediction = estimator.predict(test_X)
        correlation = r2_score(test_target, prediction)

        target_data = pd.read_csv(f'{target_path[0]}')
        shifts = target_data['Experimental']
        compound_names = target_data['Name'] # TODO: Add compound names in hold out

        residuals = [observed - pred for observed, pred in zip(test_target, prediction)]

        outliers = [(observed, pred, res) for observed, pred, res in zip(test_target, prediction, residuals) if abs(res) > threshold]

        plt.figure()
        plt.scatter(test_target, prediction, edgecolors=(0, 0, 0))
        plt.plot([test_target.min(), test_target.max()], [test_target.min(), test_target.max()], 'k--', lw=4)

        outlier_names = []

        for observed, pred, res in outliers:
            for shift, compound_name in zip(shifts, compound_names):
                if observed == shift:
                    outlier_names.append(compound_name)

            plt.scatter(observed, pred, color='red')

        print(f"Outliers ({len(outlier_names)}):\n------------")
        for (observed, pred, res), outlier_name in zip(outliers, outlier_names):
            print(f"Compound Name: {outlier_name}, Observed: {observed}, Predicted: {pred}, Residual: {res}")

        plt.xlabel('Measured [ppm]')
        plt.ylabel('Predicted [ppm]')
        plt.title(f'Parity plot ($R^2$ = {correlation:.2f})')
        plt.show()

        # TODO: Add feature to to show parity plot for holdout set

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
    # TODO: resolve optimizer divergence maybe by scaling
# TODO: Include option for noise_estimation using WhiteKernel
# TODO: add function for single predictions
# TODO: add feature to generate SOAPs from SMILES strings
# TODO: Don't forget README before re-submitting