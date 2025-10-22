import numpy as np
import os
from rdkit.Chem import rdmolfiles
from ase import Atoms
from dscribe.descriptors import SOAP
from get_atomic_props import AtomPropsDist
from base import BaseConfig

class GenDescriptors(BaseConfig):
    """Generator for molecular descriptors from XYZ structure files.

        This class extends BaseConfig to generate three types of molecular descriptors:
        - SOAP: Smooth Overlap of Atomic Positions (using the DScribe library)
        - GAPE: Gaussian Atomic Property Embedding (formerly termed 'APE-RF')
        - ChEAP: Charge and Environment Averaged Properties (formerly termed 'SIF')

        Parameters
        ----------
        config : omegaconf.DictConfig
            Hydra configuration object containing all parameters from BaseConfig.

        Notes
        -----
        XYZ files are expected to follow the naming convention: {xyz_base}{integer}.xyz
        Generated descriptors are saved as .npy files.
        """

    def __init__(self, config):
        """Initialize descriptor generator with configuration.

        Parameters
        ----------
        config : omegaconf.DictConfig
            Hydra configuration object.
        """
        super().__init__(config)


    def get_APE_RF(self, fmt='xyz', mode='all', smooth_cutoff=False, path_index=0, normalize=False):

        """Generate Gaussian Atomic Property Embedding (GAPE, formerly termed APE-RF) descriptor.

        Generates the GAPE descriptor as sum of atom-centered Gaussians weighted by
        the rescaled atomic properties (atomic_props.json) of electronegativity, atomic radius,
        and nuclear charge.
        Molecular charge is included by subtracting it from the nuclear charge of the
        central atom.

        Parameters
        ----------
        fmt : str, optional
            Input format ('xyz' for XYZ files). Default is 'xyz'.
        mode : str, optional
            'neighbors' to include only direct neighbors or 'all' to include all atoms.
            Default is 'all'.
        smooth_cutoff : bool, optional
            If True, apply smooth cutoff function. Default is False.
        path_index : int, optional
            Index for selecting path from xyz_path list. Default is 0.
        normalize : bool, optional
            If True, normalize descriptor vectors. Default is False.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, dim) containing GAPE feature vectors, where n_samples
            is the number of structures and dim is the feature vector dimensionality
            (controlled by descriptor_params['dim']).

        Notes
        -----
        Descriptor parameters expected in config:
        - descriptor_params['rcut']: float
            Radial cutoff distance
        - descriptor_params['dim']: int
            Number of grid points
        """

        APE_RF_dataset = []
        xyz_filenames = sorted(os.listdir(self.xyz_path[path_index]), key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        APE_RF_path = os.path.join(self.descriptor_path[path_index], '_'.join(str(self.descriptor_params[param]) for param in self.descriptor_params))
        os.makedirs(APE_RF_path, exist_ok=True)

        for xyz_filename in xyz_filenames:
            apd = AtomPropsDist(self.config)

            qmol = apd.get_qmol(xyz_filename, path_index)

            EN_list = apd.get_atomic_properties(target='pauling_EN', mode=mode, fmt=fmt, filename=xyz_filename, path_index=path_index)[0]
            atomic_radii_list = apd.get_atomic_properties(target='atomic_radius', mode=mode, fmt=fmt, filename=xyz_filename, path_index=path_index)[0]
            charge_list = apd.get_atomic_properties(target='nuclear_charge', mode=mode, fmt=fmt,filename=xyz_filename, path_index=path_index)[0]

            central_atom_distances = apd.get_adjacent_atoms_xyz(filename=xyz_filename, path_index=path_index)[1]
            adjacent_atom_coord_list = apd.get_adjacent_atoms_xyz(filename=xyz_filename, path_index=path_index)[4]

            central_atom_coord = apd.get_adjacent_atoms_xyz(filename=xyz_filename, path_index=path_index)[3]
            central_atom_charge = apd.get_central_atom_props(target='nuclear_charge')
            central_atom_EN = apd.get_central_atom_props(target='pauling_EN')
            central_atom_radius = apd.get_central_atom_props(target='atomic_radius')

            relative_position_vector_list = []

            for coord in adjacent_atom_coord_list:
                relative_position = central_atom_coord - coord
                relative_position_vector_list.append(relative_position)

            x = np.linspace(0.0, self.descriptor_params['rcut'], num=self.descriptor_params['dim'])

            central_gaussian = (central_atom_charge - 0.1*qmol) * np.exp((-central_atom_EN * (x - 0) ** 2) / central_atom_radius)

            if smooth_cutoff:
                smoothing = 0.5 * (np.cos((np.pi * x) / self.descriptor_params['rcut']) + 1)
            else:
                smoothing = 1

            for i in range(0, len(EN_list)):
                adjacent_gaussian = charge_list[i] * np.exp(-(EN_list[i] * (x - central_atom_distances[i]) ** 2) / atomic_radii_list[i])

                central_gaussian += adjacent_gaussian

                i += 1

            central_gaussian = central_gaussian * smoothing

            if normalize:
                central_gaussian = central_gaussian / np.linalg.norm(central_gaussian)

            APE_RF_dataset.append(central_gaussian.flatten())
            APE_RF_file = f'{int(xyz_filename.replace(self.xyz_base, "").split(".")[0])}'

            np.save(os.path.join(APE_RF_path, APE_RF_file), central_gaussian)

        return np.array(APE_RF_dataset)

    def get_APE_RF_partitioned(self):
        """Generate APE-RF descriptors for partitioned train/test datasets.

        Returns
        -------
        list of np.ndarray
            List containing [train_descriptors, test_descriptors].

        Raises
        ------
        ValueError
            If xyz_path or descriptor_path are not lists when generating partitioned data.
        """

        path_lists = [self.xyz_path, self.descriptor_path]

        for path_list in path_lists:
            if not isinstance(path_list, list):
                raise ValueError('When generating descriptors for partitioned dataset'
                                 'provide list of xyz_paths and descriptor_paths'
                                 'that correspond to the subsets of the total dataset'
                                 'that has been partitioned. Specify path to'
                                 'training data first.')
        partitioned_data = []

        for path_index in range(0, len(self.xyz_path)):
            subset = self.get_APE_RF(fmt='xyz', mode='all', path_index=path_index)

            partitioned_data.append(subset)

        return partitioned_data

    def get_total_species(self):

        """Extract all unique atomic species present in the dataset.

        Scans all XYZ files in the total dataset to identify unique atomic species.
        Required for SOAP descriptor initialization.

        Returns
        -------
        list of str
            List of unique atomic symbols present in the dataset.

        Notes
        -----
        Assumes total dataset is in directory obtained by replacing 'train_split'
        with 'total' in the first xyz_path.
        """

        path = self.xyz_path[0].replace('train_split', 'total')

        xyz_filenames = sorted(os.listdir(path),
                               key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))
        set_of_species = set()

        for xyz_filename in xyz_filenames:
            xyz_file_path = os.path.join(path, xyz_filename)

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

        return species

    def generate_SOAPs(self, path_index=0, normalize=True):

        """Generate SOAP (Smooth Overlap of Atomic Positions) descriptors.

        Generates SOAP descriptors using the DScribe library (Comput. Phys. Comm.
        247 (2020) 106949) and saves output arrays as .npy files.

        Parameters
        ----------
        path_index : int, optional
            Index for selecting path from xyz_path and descriptor_path lists.
            Default is 0.
        normalize : bool, optional
            If True, normalize SOAP vectors to unit length. Default is True.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_features) containing SOAP vectors.
            Dimensionality depends on SOAP parameters (rcut, n_max, l_max).

        Notes
        -----
        Descriptor parameters expected in config:
        - descriptor_params['rcut']: float
            Radial cutoff distance
        - descriptor_params['nmax']: int
            Number of radial basis functions
        - descriptor_params['lmax']: int
            Maximum degree of spherical harmonics
        """
        species = self.get_total_species()

        xyz_filenames = sorted(os.listdir(self.xyz_path[path_index]),
                               key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        # Setting up SOAPs with DScribe library
        SOAP_dataset = []

        descriptor_folder = '_'.join(str(self.descriptor_params[param]) for param in self.descriptor_params)
        SOAP_path = os.path.join(self.descriptor_path[path_index], descriptor_folder)

        os.makedirs(SOAP_path, exist_ok=True)

        for xyz_filename in xyz_filenames:

            xyz_file_path = os.path.join(self.xyz_path[path_index], xyz_filename)

            try:
                mol = rdmolfiles.MolFromXYZFile(xyz_file_path)
                central_atom_index = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == self.central_atom]
                atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                atom_positions = mol.GetConformer().GetPositions()

                atoms = Atoms(symbols=atom_symbols, positions=atom_positions)

                soap = SOAP(
                    species=species,
                    periodic=False,
                    r_cut=float(self.descriptor_params['rcut']),
                    n_max=int(self.descriptor_params['nmax']),
                    l_max=int(self.descriptor_params['lmax'])
                )

                soap_power_spectrum = soap.create(atoms, centers=central_atom_index)

                if normalize:
                    soap_power_spectrum = soap_power_spectrum / np.linalg.norm(soap_power_spectrum)

                SOAP_dataset.append(soap_power_spectrum.flatten())

                SOAP_file = f'{int(xyz_filename.replace(self.xyz_base, "").split(".")[0])}.npy'

                np.save(os.path.join(SOAP_path, SOAP_file), soap_power_spectrum)

            except Exception as e:
                print(e)

                pass

        return np.array(SOAP_dataset)


    def generate_SOAPs_partitioned(self):

        """Generate SOAP descriptors for partitioned train/test datasets.

        Returns
        -------
        list of np.ndarray
            List containing [train_descriptors, test_descriptors].

        Raises
        ------
        ValueError
            If xyz_path or descriptor_path are not lists when generating partitioned data.
        """

        path_lists = [self.xyz_path, self.descriptor_path]

        for path_list in path_lists:
            if not isinstance(path_list, list):
                raise ValueError('When generating descriptors for partitioned dataset '
                                 'provide list of xyz_paths and descriptor_paths that correspond to the subsets '
                                 'of the total dataset that has been partitioned. Specify path to '
                                 'training data first.')
        partitioned_data = []

        for path_index in range(0, len(self.xyz_path)):
            subset = self.generate_SOAPs(path_index)
            partitioned_data.append(subset)

        return partitioned_data

    def get_SIF(self, target_list, path_index=0, fmt='xyz', mode='neighbors', normalize=False):
        """Generate Charge and Environment Averaged Properties (ChEAP, formerly termed 'SIF')) descriptor.

        Generates the descriptor by concatenating specified mean atomic properties
        of the neighboring atoms, as well as charge and coordination number of the
        central atom into a 5D feature vector.

        Parameters
        ----------
        target_list : list of str
            List of atomic properties to include in descriptor (e.g., 'pauling_EN',
            'atomic_radius', 'nuclear_charge', 'valency', 'qmol').
        path_index : int, optional
            Index for selecting path from xyz_path list. Default is 0.
        fmt : str, optional
            Input format ('xyz' for XYZ files). Default is 'xyz'.
        mode : str, optional
            'neighbors' for bonded atoms only or 'all' for all atoms. Default is 'neighbors'.
        normalize : bool, optional
            If True, normalize descriptor vectors to unit length. Default is False.

        Returns
        -------
        list of list
            List of feature vectors, where each vector contains averaged
            atomic properties specified in target_list.
        """


        SIF_dataset = []

        xyz_filenames = sorted(os.listdir(self.xyz_path[path_index]), key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        SIF_path = os.path.join(self.descriptor_path[path_index], '_'.join(self.descriptor_params[param] for param in self.descriptor_params))

        os.makedirs(SIF_path, exist_ok=True)

        for xyz_filename in xyz_filenames:
            apd = AtomPropsDist(config=self.config)

            feature_vector = []

            for target in target_list:

                if target == 'qmol':
                    feature_vector.append(apd.get_qmol(filename=xyz_filename, path_index=path_index))

                elif target == 'valency':
                    feature_vector.append(apd.get_atomic_properties(target='pauling_EN', mode=mode, fmt=fmt,
                                                                    filename=xyz_filename, path_index=path_index)[2])

                else:
                    feature_vector.append(apd.get_atomic_properties(target=target, mode=mode, fmt=fmt,
                                                                    filename=xyz_filename, path_index=path_index)[1])

            SIF_vec = np.array(feature_vector, dtype=float)

            SIF_file = f'{int(xyz_filename.replace(self.xyz_base, "").split(".")[0])}.npy'

            np.save(os.path.join(SIF_path, SIF_file), SIF_vec)

            if normalize:
                nrm = np.linalg.norm(SIF_vec)
                if nrm > 0:
                    SIF_vec = SIF_vec / nrm

            SIF_dataset.append(SIF_vec.tolist())


        return SIF_dataset

    def get_SIF_partitioned(self, target_list):

        """Generate ChEAP descriptors for partitioned train/test datasets.

        Parameters
        ----------
        target_list : list of str
            List of atomic properties to include in descriptor.

        Returns
        -------
        list of list
            List containing [train_descriptors, test_descriptors].

        Raises
        ------
        ValueError
            If xyz_path or descriptor_path are not lists when generating partitioned data.
        """

        path_lists = [self.xyz_path, self.descriptor_path]

        for path_list in path_lists:
            if not isinstance(path_list, list):
                raise ValueError('When generating descriptors for partitioned dataset '
                                 'provide list of xyz_paths and descriptor_paths that correspond to the subsets '
                                 'of the total dataset that has been partitioned. Specify path to '
                                 'training data first.')
        partitioned_data = []

        for path_index in range(0, len(self.xyz_path)):
            subset = self.get_SIF(target_list, path_index, fmt='xyz', mode='neighbors')  # Here was a [0] 26.08. 23:16 ?
            partitioned_data.append(subset)


        return partitioned_data
