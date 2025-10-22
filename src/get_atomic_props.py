import numpy as np
import json
import statistics
from base import BaseConfig
import os


class AtomPropsDist(BaseConfig):
    """
    Utilities for interatomic analysis around a central atom.

    This class loads XYZ structures, identifies the central atom's neighbors
    using a simple van-der-Waals cutoff, and retrieves (mean) atomic property
    values from an external JSON database.

    Parameters
    ----------
    config : omegaconf.DictConfig | dict
        Configuration object consumed by :class:`BaseConfig`. Must define, at
        minimum, paths to XYZ files (``xyz_path``), the symbol of the central
        atom (``central_atom``), and the path to the atomic-properties JSON
        file (``ap_path``).

    Attributes
    ----------
    ap_path : str | pathlib.Path
        Path to the atomic properties JSON file.
    xyz_path : list[str]
        One or more directories containing XYZ files (inherited from
        :class:`BaseConfig`).
    central_atom : str
        Symbol of the central atom (inherited from :class:`BaseConfig`).

    See Also
    --------
    BaseConfig
        Parent class that provides common configuration and path handling.
    """


    def __init__(self, config):
        """
        Initialize the interatomic analysis helper.
        """
        super().__init__(config)
        self.ap_path = config.ap_path

    def get_adjacent_atoms_xyz(self, filename, path_index):

        """
        Identify neighbor atoms of the central atom from XYZ coordinates.

        Neighboring atoms are defined via a vdW-like cutoff:
        ``1.3 * (r_neighbor + r_central)`` using radii from the atomic
        properties JSON.

        Parameters
        ----------
        filename : str
            Name of the XYZ file to read.
        path_index : int
            Index into :attr:`xyz_path` selecting which directory contains
            ``filename``.

        Returns
        -------
        neighbors : list[str]
            Atomic symbols of identified neighbor atoms (order corresponds to
            ``neighbor_distance_list`` below).
        distances_to_central : list[float]
            Distances (Å) from the central atom to **all** non-central atoms in
            the molecule, in the same order as ``adjacent_atom_symbol_list``.
        adjacent_atom_symbol_list : list[str]
            Atomic symbols of **all** non-central atoms (not only neighbors).
        central_atom_coords : numpy.ndarray
            XYZ coordinates of the central atom, shape ``(3,)``.
        adjacent_atom_coords_list : list[numpy.ndarray]
            XYZ coordinates of all non-central atoms, each of shape ``(3,)``.

        Raises
        ------
        KeyError
            If an atomic symbol encountered in the XYZ file is missing from the
            atomic properties JSON.
        FileNotFoundError
            If the XYZ file cannot be found at the constructed path.
        ValueError
            If the central atom is not present in the XYZ file.

        Notes
        -----
        The returned tuple is intentionally minimal; if you need per-neighbor
        distances, you can compute them by selecting elements of
        ``distances_to_central`` at indices corresponding to entries in
        ``neighbors`` within ``adjacent_atom_symbol_list``.
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
        Read the integer molecular charge (qmol) from an XYZ file.

        The qmol value is expected on the **second** line of the XYZ file
        (following the atom count line) and will be rounded to the nearest
        integer if written as a float.

        Parameters
        ----------
        filename : str
            Name of the XYZ file to read.
        path_index : int
            Index into :attr:`xyz_path` selecting which directory contains
            ``filename``.

        Returns
        -------
        int
            Molecular charge ``qmol`` as an integer.

        Raises
        ------
        ValueError
            If the second line does not contain a numeric charge.
        FileNotFoundError
            If the XYZ file cannot be found at the constructed path.
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
        Retrieve a property value for the central atom from the JSON database.

        Parameters
        ----------
        target : {'pauling_EN', 'atomic_radius', 'nuclear_charge'}
            Name of the property to query for the central atom.

        Returns
        -------
        float
            The requested property value for the central atom.

        Raises
        ------
        ValueError
            If ``target`` is not one of the supported properties.
        Exception
            If the central atom is missing from the JSON or the property is not
            defined for that atom.
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
        prop_list : list
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
