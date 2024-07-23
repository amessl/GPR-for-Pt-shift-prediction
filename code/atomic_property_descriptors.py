
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import statistics



class AtomPropDescriptors:
    def __init__(self, ap_data_path, format, central_atom, max_valency, xyz_path=None, xyz_base=None, smiles_path=None):
        """
        Initialize class for generating atomic property descriptors ('structure independent features' and 'APE-RF').

        :param ap_data_path: path to atomic property JSON file
        :param format: file format ('smiles' or 'xyz')
        :param central_atom: symbol of the central atom ('Pt')
        :param max_valency: maximum coordination number of the central atom encountered in the dataset
        :param xyz_path: path to the xyz-file
        :param smiles_path: path to the smiles-file
        """

        self.central_atom = central_atom
        self.format = format
        self.max_valency = max_valency
        self.ap_data_path = ap_data_path
        self.xyz_path = xyz_path
        self.xyz_base = xyz_base
        self.smiles_path = smiles_path

    def get_adjacent_atoms(self):
        """
        Get direct neighbor atoms of the central atom from SMILES representation.

        :return:
        Dictionary of neighbor atoms and corresponding atomic indices,
        list of atomic indices of neighbor atoms and list of symbols of the neighbor atoms.
        """

        mol = Chem.MolFromSmiles(self.smiles_path)

        atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == self.central_atom]

        adjacent_atoms = {}
        adjacent_atoms_symbols = []
        adjacent_atoms_indices = []

        for atom_index in atom_indices:
            atom = mol.GetAtomWithIdx(atom_index)
            neighbors = atom.GetNeighbors()

        for neighbor in neighbors:
            neighbor_symbol = neighbor.GetSymbol()
            adjacent_atoms_symbols.append(neighbor_symbol)
            adjacent_atom_index = neighbor.GetIdx()
            adjacent_atoms_indices.append(adjacent_atom_index)
            if neighbor_symbol in adjacent_atoms:
                adjacent_atoms[neighbor_symbol] += 1
            else:
                adjacent_atoms[neighbor_symbol] = 1

        adjacent_atoms = {k: v for k, v in sorted(adjacent_atoms.items())}

        return adjacent_atoms, adjacent_atoms_indices, adjacent_atoms_symbols

    def get_adjacent_atoms_xyz(self):

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

        with open(self.xyz_path, 'r') as xyz_file:
            lines = xyz_file.readlines()[2:]

        central_atom_coords = []
        adjacent_atom_coords_list = []
        adjacent_atom_symbol_list = []

        for line in lines:

            if self.central_atom in line:
                elements = line.split()
                central_atom_coords = np.array([float(elements[1]), float(elements[2]), float(elements[3])])

            else:
                adjacent_elements = line.split()
                adjacent_atom_symbol = adjacent_elements[0]
                adjacent_atom_symbol_list.append(adjacent_atom_symbol)
                adjacent_atom_coords = np.array(
                    [float(adjacent_elements[1]), float(adjacent_elements[2]), float(adjacent_elements[3])])
                adjacent_atom_coords_list.append(adjacent_atom_coords)

        distance_list = []

        for coord in adjacent_atom_coords_list:
            distance = np.linalg.norm(coord - central_atom_coords)
            distance_list.append(distance)

        xyz_neighbor_list = []
        neighbor_distance_list = []

        with open(self.ap_data_path) as ap_data_file:
            ap_data = json.load(ap_data_file)

        for index, symbol in enumerate(adjacent_atom_symbol_list):
            if symbol in ap_data:
                atomic_radii_sum = ap_data[symbol]['atomic_radius'] * 1.3 + ap_data[self.central_atom][
                    'atomic_radius'] * 1.3
                atomic_radii_sum_A = atomic_radii_sum / 100

            if atomic_radii_sum_A > distance_list[index]:
                xyz_neighbor_list.append(adjacent_atom_symbol_list[index])
                neighbor_distance_list.append(distance_list[index])
                mean_distance = statistics.mean(neighbor_distance_list)


        return xyz_neighbor_list, mean_distance, neighbor_distance_list, distance_list, \
            adjacent_atom_symbol_list, central_atom_coords, adjacent_atom_coords_list

    def get_distances_smiles(self):
        """

        Get distances of the central atom to its neighbor atoms from a SMILES representation

        :return:
        mean distance of the central atom to its neighbor atoms and list of the individual distances.
        """

        mol = Chem.MolFromSmiles(self.smiles_path)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        conformer = mol.GetConformer(0)
        positions = conformer.GetPositions()

        central_atom_index = None

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == self.central_atom:
                central_atom_index = atom.GetIdx()

        distances = []

        central_atom_pos = positions[central_atom_index]

        adjacent_atoms_indices = self.get_adjacent_atoms()[1]

        for adjacent_atom_index in adjacent_atoms_indices:
            neighbor_pos = positions[adjacent_atom_index]
            distance = np.linalg.norm(np.subtract(central_atom_pos, neighbor_pos))
            distances.append(distance)
        if len(distances) < self.max_valency:
            distances.extend([0] * (self.max_valency - len(distances)))

        distances_no_zeroes = [value for value in distances if value != 0]
        mean_distance = statistics.mean(distances_no_zeroes)

        return mean_distance, distances

    def get_atomic_properties(self, target, mode):

        """
        Get atomic properties of the neighbor atoms of the central atom for a molecule.
        The atomic propeties are stored in the JSON file 'atomic_properties.json'

        :param target: Atomic property of interest ('pauling_EN' for electronegativity, 'atomic_radius',
        'nuclear_charge', 'ionization_potential', 'electron_affinity',
        'polarizability' or 'vdw_radius' for the van-der-Waals radius)
        :param mode: get atomic properties only of the neighbor atoms ('neighbors') or all atoms ('all')

        :return:
        List of the atomic properties for each atom, mean value of the property
        and coordination number of the central atom.

        """

        props = ['pauling_EN', 'atomic_radius',
                 'nuclear_charge', 'ionization_potential',
                 'electron_affinity', 'polarizability', 'vdw_radius']

        with open(self.ap_data_path) as ap_data_file:
            ap_data = json.load(ap_data_file)


        if self.format == 'xyz' and self.xyz_path is not None:

            if mode == 'neighbors':
                adjacent_atoms_list = self.get_adjacent_atoms_xyz()[0]

            elif mode == 'all':
                adjacent_atoms_list = self.get_adjacent_atoms_xyz()[4]

        elif self.format == 'smiles' and self.smiles_path is not None:
            adjacent_atoms_list = self.get_adjacent_atoms()[2]



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
            raise ValueError(f"Target property is not supported. Supported properties: \n {props}")

        valency = len(prop_list)

        return prop_list, mean_prop, valency


    def get_gaussian(self, mol_charge, mode, cutoff, dim):

        """
        Generate the APE-RF descriptor as sum of atom centered Gaussians weighted by the atomic properties
        of electronegativity, atomic radius and nuclear charge. Molecular charge is included by substracting
        it from the nuclear charge of the central atom.
        :param mol_charge: total charge of the molecule
        :param mode: generate APE-RF only up to the neighbor atoms ('neighbors') or for all atoms 'atom'
        :param cutoff: Maximum distance from the central atom in Angstrom.
        :param dim: number of values sampled from the APE-RF (dimension of the resulting feature vector)

        :return:
        1D-array of APE-RF function values
        """

        EN_list = self.get_atomic_properties(target='pauling_EN', mode=mode)[0]
        atomic_radii_list = self.get_atomic_properties(target='atomic_radius', mode=mode)[0]
        charge_list = self.get_atomic_properties(target='nuclear_charge', mode=mode)[0]
        central_atom_distances = self.get_adjacent_atoms_xyz()[3]

        central_atom_coord = self.get_adjacent_atoms_xyz()[5]
        adjacent_atom_coord_list = self.get_adjacent_atoms_xyz()[6]

        relative_position_vector_list = []

        for coord in adjacent_atom_coord_list:
            relative_position = central_atom_coord - coord
            relative_position_vector_list.append(relative_position)

        x = np.linspace(0.0, cutoff, num=dim)

        Pt_gaussian_EN = (78 - mol_charge) * np.exp(((-2.28 * (x - 0) ** 2) / (2 * 1.35)))

        for i in range(0, len(EN_list)):
            radial_EN = charge_list[i] * (
                np.exp(-(EN_list[i] * (x - central_atom_distances[i]) ** 2) / (2 * atomic_radii_list[i] / 100)))

            Pt_gaussian_EN += radial_EN

            i += 1

        return Pt_gaussian_EN
