
# import necessary libraries

import numpy as np
import os
import pandas as pd
import sklearn.svm
from dscribe.descriptors import SOAP
from rdkit.Chem import AllChem
from ase import Atoms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation, RBF
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import Normalizer
import seaborn as sns

# set up class for generating, reading and using SOAP descriptors as input for polynomial GPR

class SOAP_GPR:
    def __init__(self, SOAP_parameters, SOAP_directory = None, XYZ_directory = None, XYZ_base = None, central_atom = None):
        self.central_atom = central_atom
        self.SOAP_directory = SOAP_directory
        self.SOAP_parameters = SOAP_parameters
        self.XYZ_directory = XYZ_directory
        self.XYZ_base = XYZ_base


# function for generating SOAPs from XYZ files
    def generate_SOAPs(self):

        xyz_path = self.XYZ_directory

        xyz_filenames = sorted(os.listdir(xyz_path), key=lambda x: int(x.replace(self.XYZ_base, '').split('.')[0]))

        set_of_species = set()

        for xyz_filename in xyz_filenames:
            xyz_file_path = os.path.join(xyz_path, xyz_filename)

            try:

                if os.path.getsize(xyz_file_path) == 0:
                    raise Warning(f'XYZ file {xyz_filename} is empty')

                with open(xyz_file_path, 'r') as xyz_file:
                    lines = xyz_file.readlines()[2:]

                    for line in lines:
                        line_elements = line.split()

                        if line_elements:
                            set_of_species.add(line_elements[0])

            except Exception as e:
                print(e)

                pass

        species = list(set_of_species)
        print('Species present in dataset:', species)


    # Setting up SOAPs with DScribe library
        SOAP_dataset = []

        for xyz_filename in xyz_filenames:

            xyz_file_path = os.path.join(xyz_path, xyz_filename)

            try:
                mol = AllChem.MolFromXYZFile(xyz_file_path)
                central_atom_index = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == self.central_atom]
                atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                atom_positions = mol.GetConformer().GetPositions()

                atoms = Atoms(symbols=atom_symbols, positions=atom_positions)

                np.savetxt(f'/home/alex/ML/mol_conformer_{xyz_filename}.txt', atom_positions)


                soap = SOAP(
                    species=species,
                    periodic=False,
                    r_cut=float(self.SOAP_parameters[0]),
                    n_max=int(self.SOAP_parameters[1]),
                    l_max=int(self.SOAP_parameters[2])
                )

                soap_power_spectrum = soap.create(atoms, centers=central_atom_index)

                SOAP_dataset.append(soap_power_spectrum.flatten())

                descriptor_path = os.path.join(self.SOAP_directory,
                                f'r{self.SOAP_parameters[0]}_n{self.SOAP_parameters[1]}_l{self.SOAP_parameters[2]}/')

                os.makedirs(descriptor_path, exist_ok=True)
                SOAP_file = f'{int(xyz_filename.replace(self.XYZ_base, "").split(".")[0])}'
                #SOAP_file_list.append(SOAP_file)
                np.savetxt(f'{descriptor_path}{SOAP_file}.txt', soap_power_spectrum)

            except Exception as e:
                print(e)

                pass

        return np.array(SOAP_dataset)


# read generated SOAPs saved as txt
    def read_SOAPs(self):

        descriptor_path = os.path.join(self.SOAP_directory,
                        f'r{self.SOAP_parameters[0]}_n{self.SOAP_parameters[1]}_l{self.SOAP_parameters[2]}/')

        SOAP_dataset = []

        SOAP_filenames = sorted(os.listdir(descriptor_path), key=lambda x: int(x.split('.')[0]))
        SOAP_memory = 0
        file_count = 0

        for SOAP_filename in SOAP_filenames:
            try:
                SOAP_file = os.path.join(descriptor_path, SOAP_filename)
                SOAP_array = np.loadtxt(SOAP_file)
                #print('Norm:', np.linalg.norm(SOAP_array))
                SOAP_dataset.append(SOAP_array)
                SOAP_memory += os.path.getsize(SOAP_file)

                file_count += 1

            except os.path.getsize(SOAP_file) == 0:
                raise Warning(f'File No. {file_count} is empty.')

                pass


        print(f'SOAP files read: {len(SOAP_filenames)} \nAverage size: {round((SOAP_memory / file_count) / 1024, 3)} kB')

        return SOAP_dataset

# function for polynomial GPR taking SOAPs and response variable (chemical shifts) as input
    def predict(self, mode, regressor, kernel_degree, target_path, target_name, alpha, normalization,
                lc=None, correlation_plot=None, hypers=None, grid_search = None):

        if mode == 'read':

            X_data = self.read_SOAPs()

        elif mode == 'write':

            X_data = self.generate_SOAPs()
            print(np.shape(X_data))

        else:

            raise Exception('mode has to be specified as "read" for reading \n '
                            'already existing SOAPs or "write" for generating SOAPs.')

        target_data = pd.read_csv(f'{target_path}.csv')[str(target_name)]


        if normalization is False:

            pass

        else:
            X_data = Normalizer(norm='l2').fit_transform(X_data)

        randomSeed = 42
        train_X, test_X, train_target, test_target \
            = train_test_split(X_data, target_data, random_state=randomSeed, test_size=0.25, shuffle=True)


        if regressor == 'GPR':


            if kernel_degree == 1:
                estimator = GaussianProcessRegressor(kernel=DotProduct(),
                                                     random_state=randomSeed, alpha=float(alpha), optimizer=None)


            elif kernel_degree > 1:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(), int(kernel_degree)),
                                               random_state=randomSeed, alpha=float(alpha), optimizer=None)

            else:
                estimator =GaussianProcessRegressor(kernel=RBF(), alpha=float(alpha))

        elif regressor == 'KRR':

            if kernel_degree == 1:
                estimator = Ridge(alpha=float(alpha))

            elif kernel_degree > 1:
                estimator = KernelRidge(kernel=Exponentiation(DotProduct(), int(kernel_degree)), alpha=float(alpha))

            else:
                estimator = KernelRidge(kernel='rbf', alpha=float(alpha))


        else:
            raise Exception('Regressor type has to be specified. \n '
                            '"GPR" for Gaussian Process Regression or "KRR" for Kernel Ridge Regression')


        estimator.fit(train_X, train_target)

        cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)
        #cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
        model = estimator

        scores_rmse = cross_val_score(model, X_data, target_data, scoring='neg_root_mean_squared_error',
                                      cv=cv, n_jobs=-1)

        print('--------------------------------\nCross-validated error values:\n--------------------------------')
        print('RMSE (4-fold CV):')
        print(np.mean(np.absolute(scores_rmse)), '[ppm]')
        print(np.std(np.absolute(scores_rmse)), '[ppm]  (STDEV)')

        print('\n')

        scores_mae = cross_val_score(model, X_data, target_data, scoring='neg_mean_absolute_error',
                                     cv=cv, n_jobs=-1)
        print('MAE (4-fold CV):')
        print(np.mean(abs(scores_mae)), '[ppm]')
        print(np.std(abs(scores_mae)), '[ppm],  (STDEV)')
        print('--------------------------------')


        if lc is True:

            mae_list = []
            split_list = []

            for split in range(10, 100, 10):

                train_X, test_X, train_target, test_target \
                    = train_test_split(X_data, target_data, random_state=randomSeed, test_size=1-(split/100), shuffle=True)

                estimator.fit(train_X, train_target)
                prediction = estimator.predict(test_X)

                mae = mean_absolute_error(test_target, prediction)
                mae_list.append(mae)

                split_list.append(split)

            plt.scatter(split_list, mae_list)
            plt.plot(split_list, mae_list)
            plt.xlabel('Training data [%]')
            plt.ylabel('MAE [ppm]')
            plt.title(f'Learning Curve for Polynomial {regressor} with degree={kernel_degree} \n '
                     f'($r_{{cut}}$={float(self.SOAP_parameters[0])}, $n_{{max}}$={self.SOAP_parameters[1]}, '
                      f'$l_{{max}}$={self.SOAP_parameters[2]})')
            plt.show()

            fig, ax = plt.subplots()


            lc_plot_params = {
            "X": X_data,
            "y": target_data,
            "train_sizes": np.linspace(0.25, 1.0, 5),
            "cv": ShuffleSplit(n_splits=4, test_size=0.25, random_state=42),
            "score_type": "both",
            "n_jobs": -1,
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
            "score_name": "Correlation ($R^{2}$)",
            "scoring": "r2"
            }

            LearningCurveDisplay.from_estimator(estimator, **lc_plot_params, ax=ax)
            train_sizes, train_scores, test_scores = learning_curve(estimator, X_data, target_data, train_sizes = np.linspace(0.25, 1, 5), cv=ShuffleSplit(n_splits=4, test_size=0.25, random_state=42), scoring='neg_mean_absolute_error')
            handles, label = ax.get_legend_handles_labels()
            ax.legend(handles[:2], ["Training Score", "Test Score"], loc='lower right')

            ax.set_title('SOAP', fontsize=16)
            ax.set_ylabel('$R^{2}$', fontsize=16)
            plt.savefig(f'/home/alex/ML/SOAP_GPR_NMR/final_dataset/figures/learning_curves/'
                        f'rcut{self.SOAP_parameters[0]}_nmax{self.SOAP_parameters[1]}_lmax{self.SOAP_parameters[2]}_normalized_final.png',
                        dpi=700, bbox_inches='tight')
            plt.show()

        else:
            pass

        if correlation_plot is True:

            estimator.fit(train_X, train_target)
            prediction = estimator.predict(test_X)

            coef = np.polyfit(prediction, test_target, deg=1)
            correlation = r2_score(test_target, prediction)
            fig, ax = plt.subplots()
            print(test_target, prediction)
            ax.plot(test_target, prediction, 'go')
            ax.plot(test_target, test_target, '-g', label=f'$R^{2}$ = {np.round(correlation, 2)}')
            #f'$y = {np.round(coef[0], 4)}x - {np.round(coef[1], 3)}$ \n R = {np.round(correlation, 4)}')
            ax.set_xlabel('Experimental Shifts [ppm]')
            ax.set_ylabel('Predicted Shifts [ppm]')
            ax.set_title(f'SOAP descriptor ($r_{{cut}}$={float(self.SOAP_parameters[0])}, $n_{{max}}$={self.SOAP_parameters[1]},$l_{{max}}$={self.SOAP_parameters[2]})')
            ax.grid()
            ax.legend()

            plt.savefig(f'/home/alex/ML/SOAP_GPR_NMR/final_dataset/figures/correlation_plots/rcut{self.SOAP_parameters[0]}_nmax{self.SOAP_parameters[1]}_lmax{self.SOAP_parameters[2]}.svg', format='svg',
                        dpi=500, bbox_inches='tight')
            plt.show()


        if hypers is True:

            sigma_0_range = np.logspace(-5, 5, num=10)
            noise_level_range = np.logspace(-5, 5, num=10)

            sigma_0_grid, noise_level_grid = np.meshgrid(sigma_0_range, noise_level_range)

            log_marginal_likelihood = [estimator.log_marginal_likelihood(theta=np.log([0.36, sigma_value, noise]))
                                       for sigma_value, noise in zip(sigma_0_grid.ravel(), noise_level_grid.ravel())]

            log_marginal_likelihood = np.reshape(log_marginal_likelihood, newshape=noise_level_grid.shape)

            vmin, vmax = (-log_marginal_likelihood).min(), 10
            level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=10), decimals=1)
            plt.contour(
                sigma_0_grid,
                noise_level_grid,
                -log_marginal_likelihood,
                levels=level,
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
            plt.colorbar()
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Sigma 0")
            plt.ylabel("Noise-level")
            plt.title("Log-marginal-likelihood")
            plt.savefig('/home/alex/ML/SOAP_GPR_NMR/final_dataset/figures/hyperparameters_dot_product_kernel_minima.png', dpi=400)
            plt.show()

        else:
            pass

        if grid_search is True:

            param_grid = {  # Possible values for descriptor parameter 3
                'alpha': [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]}  # Regularization strength

            cv = KFold(n_splits=4, shuffle=True, random_state=42)

            # Define GridSearchCV
            grid_search = RandomizedSearchCV(estimator, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, n_iter=100)

            # Fit the grid search to your data
            grid_search.fit(train_X, train_target)  # X_train should contain your training data and y_train your target values

            # Get the best parameters and the corresponding mean absolute error
            best_params = grid_search.best_params_
            best_mae = -grid_search.best_score_

        else:
            pass

        return np.mean(np.abs(scores_mae)), np.std(np.abs(scores_mae)), np.mean(np.abs(scores_rmse)), np.std(
            np.abs(scores_rmse))#, train_sizes, test_scores


def exhaustive_search(rcut_list, nmax_list, lmax_list, alpha_list):

    SOAP_directory = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/SOAPs/'

    XYZ_directory = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/xyz_files_final_set/'
    XYZ_base = 'st_'

    target_name = 'Experimental'
    target_path = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/Pt_II_complexes_final'


    mae_list = []
    parameter_combination_list = []

    for rcut in rcut_list:
        for nmax in nmax_list:
            for lmax in lmax_list:
                    for alpha in alpha_list:


                        try:

                            SOAP_ML = SOAP_GPR(SOAP_parameters=[rcut, nmax, lmax], SOAP_directory=SOAP_directory, XYZ_directory=XYZ_directory,
                            XYZ_base=XYZ_base, central_atom='Pt')

                            errors_std = SOAP_ML.predict(mode='read', regressor='GPR', kernel_degree=5, target_path=target_path,
                                         target_name=target_name, alpha=alpha, normalization=False)

                            mae_list.append(errors_std[0])
                            parameter_combination_list.append([rcut, nmax, lmax, alpha])

                            np.savetxt(f'/home/alex/ML/SOAP_GPR_NMR/final_dataset/prediction_errors/polynomial_kernel_unnormalized/rcut{int(rcut)}_nmax{nmax}_lmax{lmax}_alpha{alpha}_degree5_unnormalized.txt', np.array(errors_std), delimiter=',')

                        except Exception as e:
                            print(e)
                            pass

    mae_parameters_combined = [sublist + [item] for sublist, item in zip(parameter_combination_list, mae_list)]
    np.savetxt('/home/alex/ML/SOAP_GPR_NMR/final_dataset/prediction_errors/polynomial_kernel_unnormalized/mae_parameters_combined_unnormalized_degree5.txt', np.array(mae_parameters_combined), delimiter=',')

    best_params, min_error = min(zip(parameter_combination_list, mae_list), key=lambda x: x[1])

    print(best_params, min_error)

    return best_params, min_error


