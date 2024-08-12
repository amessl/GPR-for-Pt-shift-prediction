
import numpy as np
import os
from rdkit.Chem import AllChem
from ase import Atoms
from dscribe.descriptors import SOAP
from get_atomic_props import atom_props_dist


class generate_descriptors:

    def __init__(self, descriptor_params, descriptor_path, central_atom, xyz_path, xyz_base):
        self.descriptor_params = descriptor_params
        self.descriptor_path = descriptor_path
        self.central_atom = central_atom
        self.xyz_path = xyz_path
        self.xyz_base = xyz_base
    def get_APE_RF(self, format='xyz', mode='all', save=True):

        """
        Generate the APE-RF descriptor as sum of atom centered Gaussians weighted by the atomic properties
        of electronegativity, atomic radius and nuclear charge. Molecular charge is included by substracting
        it from the nuclear charge of the central atom.
        ape_rf_params = [mol_charge, cutoff, dim]
        :param mol_charge: total charge of the molecule
        :param mode: generate APE-RF only up to the neighbor atoms ('neighbors') or for all atoms ('all')
        :param cutoff: Maximum distance from the central atom in Angstrom.
        :param dim: number of values sampled from the APE-RF (dimension of the resulting feature vector)

        Order of descriptor params: [q_mol, r_cut, dim]

        :return:
        1D-array of APE-RF function values
        """

        APE_RF_dataset = []
        xyz_filenames = sorted(os.listdir(self.xyz_path), key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        APE_RF_path = os.path.join(self.descriptor_path, '_'.join(str(param) for param in self.descriptor_params))

        os.makedirs(APE_RF_path, exist_ok=True)

        for xyz_filename in xyz_filenames:
            apd = atom_props_dist(central_atom=self.central_atom, xyz_base=self.xyz_base,
                                  xyz_path=os.path.join(self.xyz_path, xyz_filename))

            qmol = apd.get_qmol()

            EN_list = apd.get_atomic_properties(target='pauling_EN', mode=mode, format=format)[0]
            atomic_radii_list = apd.get_atomic_properties(target='atomic_radius', mode=mode, format=format)[0]
            charge_list = apd.get_atomic_properties(target='nuclear_charge', mode=mode, format=format)[0]

            central_atom_distances = apd.get_adjacent_atoms_xyz()[3]
            adjacent_atom_coord_list = apd.get_adjacent_atoms_xyz()[6]

            central_atom_coord = apd.get_adjacent_atoms_xyz()[5]
            central_atom_charge = apd.get_central_atom_props(target='nuclear_charge')
            central_atom_EN = apd.get_central_atom_props(target='pauling_EN')
            central_atom_radius = apd.get_central_atom_props(target='atomic_radius')

            relative_position_vector_list = []

            for coord in adjacent_atom_coord_list:
                relative_position = central_atom_coord - coord
                relative_position_vector_list.append(relative_position)

            x = np.linspace(0.0, self.descriptor_params[0], num=self.descriptor_params[1])

            central_gaussian = (central_atom_charge - qmol) * np.exp(((-central_atom_EN * (x - 0) ** 2) / (2 * (central_atom_radius/100))))

            for i in range(0, len(EN_list)):
                adjacent_gaussian = charge_list[i] * (
                    np.exp(-(EN_list[i] * (x - central_atom_distances[i]) ** 2) / (2 * atomic_radii_list[i] / 100)))

                central_gaussian += adjacent_gaussian

                i += 1

            APE_RF_dataset.append(central_gaussian.flatten())
            APE_RF_file = f'{int(xyz_filename.replace(self.xyz_base, "").split(".")[0])}.txt'

            if save:
                np.save(os.path.join(APE_RF_path, APE_RF_file), central_gaussian)

        return APE_RF_dataset

    def generate_SOAPs(self, save=True):

        xyz_filenames = sorted(os.listdir(self.xyz_path), key=lambda x: int(x.replace(self.xyz_base, '').split('.')[0]))

        set_of_species = set()

        for xyz_filename in xyz_filenames:
            xyz_file_path = os.path.join(self.xyz_path, xyz_filename)

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

        descriptor_folder = '_'.join([str(param) for param in self.descriptor_params])
        SOAP_path = os.path.join(self.descriptor_path, descriptor_folder)

        os.makedirs(SOAP_path, exist_ok=True)

        for xyz_filename in xyz_filenames:

            xyz_file_path = os.path.join(self.xyz_path, xyz_filename)

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
                    r_cut=float(self.descriptor_params[0]),
                    n_max=int(self.descriptor_params[1]),
                    l_max=int(self.descriptor_params[2])
                )

                soap_power_spectrum = soap.create(atoms, centers=central_atom_index)

                SOAP_dataset.append(soap_power_spectrum.flatten())

                SOAP_file = f'{int(xyz_filename.replace(self.xyz_base, "").split(".")[0])}.npy'

                if save:
                    np.save(os.path.join(SOAP_path, SOAP_file), soap_power_spectrum)

            except Exception as e:
                print(e)

                pass

        return np.array(SOAP_dataset)


# TODO: Modify descriptor generation to enable generation of single descriptors