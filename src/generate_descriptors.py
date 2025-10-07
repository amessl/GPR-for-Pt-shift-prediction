import numpy as np
import os
from rdkit.Chem import rdmolfiles
from ase import Atoms
from dscribe.descriptors import SOAP
from get_atomic_props import AtomPropsDist
from base import BaseConfig

class GenDescriptors(BaseConfig):

    def __init__(self, config):

        """
        Initialize class for generating descriptors (currently only APE_RF and SOAP are supported),
        Descriptor parameters have to be specified either for SOAP or APE_RF individually when
        creating an instance of this class. xyz_files are assumed to consist of some basename
        followed by an integer number ('xyz_base_int.xyz')
        """
        super().__init__(config)


    def get_APE_RF(self, format='xyz', mode='all', smooth_cutoff=False, path_index=0, normalize=False):

        """
        Generate the APE-RF descriptor as sum of atom centered Gaussians weighted by the atomic properties
        of electronegativity, atomic radius and nuclear charge. Molecular charge is included by substracting
        it from the nuclear charge of the central atom.

        :param format: Whether to read structure from SMILES-file (currently not supported) or xyz_file (default)
        :param mode: generate APE-RF only up to the neighbor atoms ('neighbors') or for all atoms (default)
        Order of descriptor params: [r_cut, dim]

        :return:
        n x p-array of discretized APE_RF-values (APE_RF-vectors), where n is the number of samples (structures)
        and p the dimensionality of each APE_RF-vector (controlled by the second APE_RF-parameter)
        """

        APE_RF_dataset = []
        xyz_filenames = sorted(os.listdir(self.xyz_path[path_index]), key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        APE_RF_path = os.path.join(self.descriptor_path[path_index], '_'.join(str(param) for param in self.descriptor_params))

        os.makedirs(APE_RF_path, exist_ok=True)

        for xyz_filename in xyz_filenames:
            apd = AtomPropsDist(self.config)

            qmol = apd.get_qmol(xyz_filename, path_index)

            EN_list = apd.get_atomic_properties(target='pauling_EN', mode=mode, format=format, filename=xyz_filename, path_index=path_index)[0]
            atomic_radii_list = apd.get_atomic_properties(target='atomic_radius', mode=mode, format=format, filename=xyz_filename, path_index=path_index)[0]
            charge_list = apd.get_atomic_properties(target='nuclear_charge', mode=mode, format=format,filename=xyz_filename, path_index=path_index)[0]

            central_atom_distances = apd.get_adjacent_atoms_xyz(filename=xyz_filename, path_index=path_index)[3]
            adjacent_atom_coord_list = apd.get_adjacent_atoms_xyz(filename=xyz_filename, path_index=path_index)[6]

            central_atom_coord = apd.get_adjacent_atoms_xyz(filename=xyz_filename, path_index=path_index)[5]
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
            subset = self.get_APE_RF(format='xyz', mode='all', path_index=path_index)

            partitioned_data.append(subset)

        return partitioned_data

    def get_total_species(self):

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

        """
        Generate the SOAP-descriptor using the DScribe-library (Comput. Phys. Comm. 247 (2020) 106949)
        and save the output array as .npy-file. Descriptor parameters are specified when creating an
        instance of this class. Order of SOAP-parameters: [r_cut, n_max, l_max].

        :param path_index: List index of path to structures and descriptors
                           (Default is changed in partitioned method)

        :return:
        n x p-array of SOAP-vectors, where n is the number of samples (structures) and p the
        dimensionality of each SOAP-vector (dimensionality can be very high depending on
        the settings of the SOAP-parameters).
        """
        species = self.get_total_species()

        xyz_filenames = sorted(os.listdir(self.xyz_path[path_index]),
                               key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        # Setting up SOAPs with DScribe library
        SOAP_dataset = []

        descriptor_folder = '_'.join([str(param) for param in self.descriptor_params])
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

    def get_SIF(self, target_list, path_index=0, format='xyz', mode='neighbors', normalize=False):

        SIF_dataset = []

        xyz_filenames = sorted(os.listdir(self.xyz_path[path_index]), key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        SIF_path = os.path.join(self.descriptor_path[path_index], '_'.join(str(param) for param in self.descriptor_params))

        os.makedirs(SIF_path, exist_ok=True)

        for xyz_filename in xyz_filenames:
            apd = AtomPropsDist(config=self.config)

            feature_vector = []

            for target in target_list:

                if target == 'qmol':
                    feature_vector.append(apd.get_qmol(filename=xyz_filename, path_index=path_index))

                elif target == 'valency':
                    feature_vector.append(apd.get_atomic_properties(target='pauling_EN', mode=mode, format=format,
                                                                    filename=xyz_filename, path_index=path_index)[2])

                else:
                    feature_vector.append(apd.get_atomic_properties(target=target, mode=mode, format=format,
                                                                    filename=xyz_filename, path_index=path_index)[1])

            SIF_vec = np.array(feature_vector, dtype=float)

            if normalize:
                nrm = np.linalg.norm(SIF_vec)
                if nrm > 0:
                    SIF_vec = SIF_vec / nrm

            SIF_dataset.append(SIF_vec.tolist())

        return SIF_dataset

    def get_SIF_partitioned(self, target_list):

        path_lists = [self.xyz_path, self.descriptor_path]

        for path_list in path_lists:
            if not isinstance(path_list, list):
                raise ValueError('When generating descriptors for partitioned dataset '
                                 'provide list of xyz_paths and descriptor_paths that correspond to the subsets '
                                 'of the total dataset that has been partitioned. Specify path to '
                                 'training data first.')
        partitioned_data = []

        for path_index in range(0, len(self.xyz_path)):
            subset = self.get_SIF(target_list, path_index, format='xyz', mode='neighbors')  # Here was a [0] 26.08. 23:16 ?
            partitioned_data.append(subset)


        return partitioned_data
