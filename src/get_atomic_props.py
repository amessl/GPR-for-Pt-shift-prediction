import numpy as np
import json
import statistics
from base import BaseConfig
import os


class AtomPropsDist(BaseConfig):
    def __init__(self, config):
        """
        Initialize class for getting interatomic distances, neighbors of central atom and (mean) atomic properties

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

        props = ['pauling_EN', 'atomic_radius', 'nuclear_charge']

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

    def get_atomic_properties(self, fmt, target, mode, filename, path_index):

        """
            Get atomic property values for neighbor or all atoms around the central atom.

            Parameters
            ----------
            fmt : {'xyz'}
                Structure fmt.
            target : str
                One of: 'pauling_EN', 'atomic_radius', 'nuclear_charge',
            mode : {'neighbors','all'}
                Whether to use only neighbor atoms or all atoms (excluding the central atom).
            filename : str
                XYZ file name.
            path_index : int
                Index into self.xyz_path.

            Returns
            -------
            prop_list : list[float]
                Property values in the same order as selected atoms.
            mean_prop : float
                Mean of the property values (NaN if no atoms selected).
            valency : int
                Number of atoms used (neighbors if mode='neighbors', else all non-central atoms).
            """

        allowed_props = ['pauling_EN', 'atomic_radius', 'nuclear_charge']

        if target not in allowed_props:
            raise ValueError(f"Target property '{target}' is not supported. Supported: {sorted(allowed_props)}")

        with open(self.ap_path) as ap_data_file:
            ap_data = json.load(ap_data_file)

        if fmt == 'xyz' and self.xyz_path is not None:

            if mode == 'neighbors':
                adjacent_atoms_list = self.get_adjacent_atoms_xyz(filename, path_index)[0]

            elif mode == 'all':
                adjacent_atoms_list = self.get_adjacent_atoms_xyz(filename, path_index)[2]

            else:
                raise ValueError(f"Unknown mode '{mode}'. Supported: 'all' or 'neighbors'.")

        else:
            raise ValueError(f"Unknown fmt: {fmt}. Supported: 'xyz'")


        prop_list = [ap_data[atom_symb][target] for atom_symb in adjacent_atoms_list]
        mean_prop = statistics.mean(prop_list)

        valency = len(prop_list)

        return prop_list, mean_prop, valency
