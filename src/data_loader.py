
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Normalizer
from base import BaseConfig
from omegaconf import OmegaConf
from generate_descriptors import GenDescriptors


class DataLoader(BaseConfig):

    """Data loader for inputs (molecular descriptors) and labels (a.k.a. target values)
    for the GPR model.

    This class extends BaseConfig to load molecular descriptors
    and the experimental chemical shifts. Supports both
    reading pre-generated descriptors and generating them on-the-fly.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Hydra configuration object containing:
        - All parameters from BaseConfig
        - config.mode : str
            'read' for loading pre-generated descriptors or 'write' for generating new ones
        - config.backend.model.target_path : str or list of str
            Path(s) to CSV files containing target values

    Attributes
    ----------
    mode : str
        Operating mode ('read' or 'write').
    target_path : list of str
        List of paths to CSV files containing the labels.

    Raises
    ------
    ValueError
        If target_path is not a string or list of strings.
        If descriptor files are empty.
    """

    def __init__(self, config):
        """Initialize DataLoader with configuration.

        Parameters
        ----------
        config : omegaconf.DictConfig
            Hydra configuration object.
        """

        super().__init__(config)
        self.mode = config.mode

        target_path = config.backend.model.target_path

        if isinstance(target_path, str):
            self.target_path = [target_path]
        elif OmegaConf.is_list(target_path):
            self.target_path = list(target_path)
        else:
            raise ValueError("Paths should be a string or a list of strings")

    def read_descriptors(self, path_index):

        """Read pre-generated descriptors from the corresponding directory.

        Loads molecular descriptors from .npy files in the specified directory.
        Optionally applies L2 normalization if configured.

        Parameters
        ----------
        path_index : int
            Index for selecting the path when data is partitioned into train/test (holdout) sets.

        Returns
        -------
        np.ndarray
            Design matrix of shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If any descriptor file is empty.
        """

        dataset = []

        descriptor_path = os.path.join(self.descriptor_path[path_index],
                                       '_'.join(str(param) for param in self.descriptor_params))
        descriptor_filenames = sorted(os.listdir(descriptor_path), key=lambda x: int(x.split('.')[0]))

        memory = 0
        file_count = 0

        for filename in descriptor_filenames:

            descriptor_file = os.path.join(descriptor_path, filename)

            if os.path.getsize(descriptor_file) == 0:
                raise ValueError(f"Descriptor file {descriptor_file} is empty.")

            descriptor_array = np.load(descriptor_file, allow_pickle=True)
            descriptor_array = descriptor_array.flatten()

            dataset.append(descriptor_array)
            memory += os.path.getsize(descriptor_file)

            file_count += 1

        if self.config.normalize:
            dataset = Normalizer(norm='l2').fit_transform(dataset)

        print(f'Descriptor files read: {len(descriptor_filenames)} \nAverage size: '
              f'{round((memory / file_count) / 1024, 3)} kB \n')

        print(f'Dimensions of design matrix: {np.shape(dataset)}')

        return dataset

    def load_samples(self, partitioned=True):
        """Load or generate molecular descriptors from XYZ structures.

        Loads descriptors from files (read mode) or generates them on-the-fly (write mode).
        Supports partitioned data (separate train/test (holdout)) or full dataset.

        Parameters
        ----------
        partitioned : bool, optional
            If True, load train and test samples separately. Default is True.

        Returns
        -------
        X_data : np.ndarray or list
            Training feature matrix or full dataset.
        X_holdout : np.ndarray or None
            Test feature matrix if partitioned, else None.

        Raises
        ------
        Exception
            If mode is not 'read' or 'write'.
            If the descriptor type is invalid.
        """

        gen = GenDescriptors(config=self.config)

        X_holdout = None

        if self.mode == 'read':

            X_data = []

            for path_index in range(0, len(self.descriptor_path)):
                X_data.append(self.read_descriptors(path_index))

            if partitioned:
                X_train, X_holdout = X_data
            else:
                X_data = np.vstack(X_data) if len(X_data) > 1 else X_data[0]


        elif self.mode == 'write':

            if self.descriptor_type == 'SOAP':

                if partitioned:
                    X_data, X_holdout = gen.generate_SOAPs_partitioned()
                else:
                    X_data = gen.generate_SOAPs()

            elif self.descriptor_type == 'GAPE':

                if partitioned:
                    X_data, X_holdout = gen.get_APE_RF_partitioned()

                else:
                    X_data = gen.get_APE_RF(path_index=2) # TODO: remove training on whole set for all (not needed)

            elif self.descriptor_type == 'ChEAP':

                if partitioned:
                    X_data, X_holdout = gen.get_SIF_partitioned(target_list=self.descriptor_params)

                else:
                    X_data = gen.get_SIF(target_list=self.descriptor_params, path_index=2)

            else:
                raise Exception('Descriptor type has to be specified. Use "SOAP" or "GAPE"')

        else:
            raise Exception('Mode has to be specified as "read" for using pre-generated descriptors \n'
                            'or "write" for generating new descriptors and passing them as input"')

        return X_data, X_holdout

    def load_targets(self, target_name='Experimental', partitioned=True):
        """Load target chemical shift values from CSV files.

           Reads experimental labels (chemical shifts) from CSV files.
           Supports partitioned data with separate train/test (holdout) files.

           Parameters
           ----------
           target_name : str, optional
               Column name in CSV file containing target values. Default is 'Experimental'.
           partitioned : bool, optional
               If True, load train and test targets separately. Default is True.

           Returns
           -------
           target_data : pd.Series
               Training set labels or full dataset.
           target_holdout : pd.Series or None
               Test (Holdout) Set labels if partitioned, else None.
           indices : pd.Series
               Sample indices from the dataset.
           target_holdout_compound_names : pd.Series or None
               Compound names for the holdout test set if available, else None.
           """
        target_holdout_compound_names = None

        if partitioned:

            target_training_data = pd.read_csv(f'{self.target_path[0]}')
            target_test_data = pd.read_csv(f'{self.target_path[1]}')

            sorted_train = target_training_data.sort_values(by='Index')
            sorted_test = target_test_data.sort_values(by='Index')

            target_data = sorted_train[str(target_name)]
            target_holdout = sorted_test[str(target_name)]

            try:
                target_holdout_compound_names = sorted_test['Name']

            except KeyError as key_err:
                print('Error encountered while reading compound names. \n',
                      f'No key {key_err} in target data.')
                print('Proceeding.')

                target_holdout_compound_names = None

        else:
            indexed_target_data = pd.read_csv(f'{self.target_path[2]}')
            sorted_train = indexed_target_data.sort_values(by='Index')

            target_data = sorted_train[str(target_name)]
            target_holdout = None

        return target_data, target_holdout, sorted_train['Index'], target_holdout_compound_names
