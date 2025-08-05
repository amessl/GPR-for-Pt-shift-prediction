
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Normalizer
from base import BaseConfig
from omegaconf import OmegaConf
from generate_descriptors import GenDescriptors


class DataLoader(BaseConfig):

    def __init__(self, config):
        """
       Initialize class for using descriptors as input for Gaussian Process Regression
       with cross-validated errors, hyperparameter tuning and visualization of learning curves

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

        if self.config.normalize:
            dataset = Normalizer(norm='l2').fit_transform(dataset)

        print(f'Descriptor files read: {len(descriptor_filenames)} \nAverage size: '
              f'{round((memory / file_count) / 1024, 3)} kB \n')

        print(f'Dimensions of design matrix: {np.shape(dataset)}')

        return dataset

    def load_samples(self, partitioned=True):
        """
        Loads samples (representations a. k. a. feature vectors) from pre-generated files
        or generating them, depending on the "mode" attribute ("read" or "write")

        :param partitioned: Whether to load train and test samples separately (Default=True)
        :return: Design matrix of total dataset or training and test samples (holdout)
        """

        gen = GenDescriptors(config=self.config)

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
                    X_data = gen.generate_SOAPs_partitioned()[0]
                    X_holdout = gen.generate_SOAPs_partitioned()[1]
                else:
                    X_data = gen.generate_SOAPs()

            elif self.descriptor_type == 'GAPE':

                if partitioned:
                    X_data = gen.get_APE_RF_partitioned()[0]
                    X_holdout = gen.get_APE_RF_partitioned()[1]

                else:
                    X_data = gen.get_APE_RF()

            elif self.descriptor_type == 'ChEAP':

                if partitioned:
                    X_data = gen.get_SIF_partitioned(target_list=self.descriptor_params)[0]
                    X_holdout = gen.get_SIF_partitioned(target_list=self.descriptor_params)[1]

                else:
                    X_data = gen.get_SIF(target_list=self.descriptor_params)

            else:
                raise Exception('Descriptor type has to be specified. Use "SOAP" or "GAPE"')

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

            try:
                target_holdout_compound_names = sorted_test['Name']

            except KeyError as key_err:
                print('Error encountered while reading compound names. \n',
                      f'No key {key_err} in target data.')
                print('Proceeding.')

                target_holdout_compound_names = None

        else:
            indexed_target_data = pd.read_csv(f'{self.target_path[0]}')
            sorted_train = indexed_target_data.sort_values(by='Index')

            target_data = sorted_train[str(target_name)]
            target_holdout = None

        return target_data, target_holdout, sorted_train['Index'], target_holdout_compound_names