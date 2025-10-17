import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import statistics
from base import BaseConfig
import os


class AtomPropsDist(BaseConfig):
    def __init__(self, config):
        """
        Initialize class for getting interatomic distances, neighbors of central atom and (mean) atomic properties

        :param smiles_path: path to the smiles-file
        """
        super().__init__(config)
        self.ap_path = config.ap_path


    def get_adjacent_atoms_xyz(self, filename, path_index):

        """
        Get direct neighbor atoms of the central atom from cartesian (XYZ) atomic coordinates.

        :return:
        List of neighbor atoms, mean distance of the central atom to the neighbor atoms,
        list of the distances of the central atom to its neighbor atoms,
        list of distances of the central atom to all other atoms,
        list of atomic symbols of the neighbor atoms,
        XYZ coordinates of the central atom,
        list of XYZ coordinates of the neighbor atoms
        """
        path = os.path.join(self.xyz_path[path_index], filename)

        with open(path, 'r') as xyz_file:
            lines = xyz_file.readlines()[2:]

        central_atom_coords = []
        adjacent_atom_coords_list = []
        adjacent_atom_symbol_list = []

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            symbol = parts[0]

            if symbol == self.central_atom:
                central_atom_coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

            else:
                adjacent_atom_symbol_list.append(symbol)
                adjacent_atom_coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                adjacent_atom_coords_list.append(adjacent_atom_coords)

        distance_list = []

        for coord in adjacent_atom_coords_list:
            distance = np.linalg.norm(coord - central_atom_coords)
            distance_list.append(distance)

        xyz_neighbor_list = []
        xyz_neighbor_set = set()
        neighbor_distance_list = []


        with open(self.ap_path) as ap_data_file:
            ap_data = json.load(ap_data_file)

        for index, symbol in enumerate(adjacent_atom_symbol_list):
            if symbol in ap_data:
                atomic_radii_sum = ap_data[symbol]['atomic_radius'] * 1.3 + ap_data[self.central_atom][
                    'atomic_radius'] * 1.3
                atomic_radii_sum_A = atomic_radii_sum

                if atomic_radii_sum_A > distance_list[index]:
                    xyz_neighbor_list.append(adjacent_atom_symbol_list[index])
                    xyz_neighbor_set.add(adjacent_atom_symbol_list[index])
                    neighbor_distance_list.append(distance_list[index])

            else:
                raise KeyError(f"Symbol '{symbol}' not found in atomic properties data.")


        return (xyz_neighbor_list, distance_list, adjacent_atom_symbol_list,
                central_atom_coords, adjacent_atom_coords_list)


    def get_qmol(self, filename, path_index):

        """
        Obtain the molecular charge (qmol) from the xyz-file.
        The qmol value ($qmol \in \mathbb{Z}$) is assumed to be
        included in the second line of each xyz-file

        :return:
        Integer value of molecular charge (qmol)
        """
        path = os.path.join(self.xyz_path[path_index], filename)
        with open(path, 'r') as xyz_file:
            qmol_line = xyz_file.readlines()[1]
            qmol = qmol_line.strip()

        try:
            int_qmol = int(round(float(qmol)))

        except ValueError:
            raise ValueError("Molecular charge not found in xyz-file. \n"
                             "Value has to be included in second line of xyz-file")
        return int_qmol

    def get_central_atom_props(self, target):

        """
        Get the atomic properties from atomic_props.json only for the central
        atom as specified when creating an instance of the AtomPropsDist class.

        :param target: Name (str) of the target property to obtain for the central atom

        :return: Value of the atomic property
        """

        props = ['pauling_EN', 'atomic_radius',
                 'nuclear_charge', 'ionization_potential',
                 'electron_affinity', 'polarizability', 'vdw_radius']

        with open(self.ap_path) as ap_data_file:
            ap_data = json.load(ap_data_file)

        atom_symbol = self.central_atom

        if target not in props:
            raise ValueError(f"Target property is not supported. Supported properties: {props}")

        if atom_symbol not in ap_data:
            raise Exception(f"Central atom {atom_symbol} not included in atomic properties JSON file.")

        atomic_property = ap_data[atom_symbol].get(target, None)
        if atomic_property is None:
            raise Exception(f"Property {target} not found for central atom {atom_symbol}.")

        return atomic_property

    def get_atomic_properties(self, format, target, mode, filename, path_index):

        """
        Get atomic properties of the neighbor atoms of the central atom for a molecule.
        The atomic properties are stored in the JSON file 'atomic_props.json'

        :param format: Whether to use xyz- or SMILES-files to read the structure
                       (Specify as 'xyz' or 'smiles')
        :param target: Atomic property of interest ('pauling_EN' for electronegativity,
                       'atomic_radius', 'nuclear_charge', 'ionization_potential',
                       'electron_affinity','polarizability' or 'vdw_radius'
                        for the van-der-Waals radius)
        :param mode: get atomic properties only of the neighbor atoms ('neighbors') or
                     all atoms ('all')

        :return:
        List of the atomic properties for each atom, mean value of the property
        and coordination number of the central atom.

        """

        props = ['pauling_EN', 'atomic_radius',
                 'nuclear_charge', 'ionization_potential',
                 'electron_affinity', 'polarizability', 'vdw_radius']

        with open(self.ap_path) as ap_data_file:
            ap_data = json.load(ap_data_file)

        if format == 'xyz' and self.xyz_path is not None:

            if mode == 'neighbors':
                adjacent_atoms_list = self.get_adjacent_atoms_xyz(filename, path_index)[0]

            elif mode == 'all':
                adjacent_atoms_list = self.get_adjacent_atoms_xyz(filename, path_index)[2]

        elif format == 'smiles' and self.smiles_path is not None:
            adjacent_atoms_list = self.get_adjacent_atoms_smiles()[2]

        else:
            raise ValueError("Specify 'format' as either 'xyz' or 'smiles'")

        prop_list = []

        if target == props[0]:

            for atom_symbol in adjacent_atoms_list:
                if atom_symbol in ap_data:
                    atomic_property = ap_data[atom_symbol][props[0]]
                    prop_list.append(atomic_property)

                    if not adjacent_atoms_list:
                        prop_list.append(0)
                else:
                    raise Exception(f"Neighboring atom {atom_symbol} not included in atomic properties JSON file.")

            mean_prop = statistics.mean(prop_list)

        elif target == props[1]:
            for atom_symbol in adjacent_atoms_list:
                if atom_symbol in ap_data:
                    atomic_property = ap_data[atom_symbol][props[1]]
                    prop_list.append(atomic_property)
                else:
                    raise Exception(f"Neighboring atom '{atom_symbol}' not included in atomic properties JSON file.")
            mean_prop = statistics.mean(prop_list)

        elif target == props[2]:
            for atom_symbol in adjacent_atoms_list:
                if atom_symbol in ap_data:
                    atomic_property = ap_data[atom_symbol][props[2]]
                    prop_list.append(atomic_property)
                else:
                    raise Exception(f"Neighboring atom {atom_symbol} not included in atomic properties JSON file.")

            mean_prop = statistics.mean(prop_list)

        elif target == props[3]:
            for atom_symbol in adjacent_atoms_list:
                if atom_symbol in ap_data:
                    atomic_property = ap_data[atom_symbol][props[3]]
                    prop_list.append(atomic_property)
                else:
                    raise Exception(f"Neighboring atom {atom_symbol} not included in atomic properties JSON file.")

            mean_prop = statistics.mean(prop_list)

        elif target == props[4]:
            for atom_symbol in adjacent_atoms_list:
                if atom_symbol in ap_data:
                    atomic_property = ap_data[atom_symbol][props[4]]
                    prop_list.append(atomic_property)
                else:
                    raise Exception(f"Neighboring atom {atom_symbol} not included in atomic properties JSON file.")

            mean_prop = statistics.mean(prop_list)

        elif target == props[5]:
            for atom_symbol in adjacent_atoms_list:
                if atom_symbol in ap_data:
                    atomic_property = ap_data[atom_symbol][props[5]]
                    prop_list.append(atomic_property)
                else:
                    raise Exception(f"Neighboring atom {atom_symbol} not included in atomic properties JSON file.")

            mean_prop = statistics.mean(prop_list)

        elif target == props[6]:
            for atom_symbol in adjacent_atoms_list:
                if atom_symbol in ap_data:
                    atomic_property = ap_data[atom_symbol][props[6]]
                    prop_list.append(atomic_property)
                else:
                    raise Exception(f"Neighboring atom {atom_symbol} not included in atomic properties JSON file.")

            mean_prop = statistics.mean(prop_list)

        else:
            raise ValueError(f"Target property '{target}' is not supported. Supported properties: \n {props}")

        valency = len(prop_list)

        return prop_list, mean_prop, valency
