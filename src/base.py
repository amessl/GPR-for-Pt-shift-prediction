
from omegaconf import OmegaConf


class BaseConfig:
    """Base configuration class for descriptor paths and parameters.

    This class initializes and validates configuration parameters for molecular
    descriptors and coordinate file paths. It supports the types
    of descriptors used in our publication: ChEAP, GAPE, and SOAP.
    Configuration is managed through the Hydra framework, which uses OmegaConf
    for hierarchical configuration management.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Hydra configuration object (DictConfig) containing descriptor parameters
        and file paths. This is provided by Hydra's @hydra.main decorator
        from YAML configuration files.
        Expected structure:
        - config.representations.rep : str
            Type of descriptor ('ChEAP', 'GAPE', or 'SOAP')
        - config.representations.{descriptor_type}_paths : str or list of str
            Path(s) to descriptor files
        - config.representations.{descriptor_type}_params : dict
            Parameters specific to the descriptor type
        - config.central_atom : str or int
            Specification of the central atom ('Pt' in this work)
        - config.xyz_dir : str or list of str
            Path(s) to XYZ coordinate files
        - config.xyz_base : str
            Base path for XYZ files

    Attributes
    ----------
    config : omegaconf.DictConfig
        The full Hydra configuration object.
    descriptor_type : str
        Type of molecular descriptor ('ChEAP', 'GAPE', or 'SOAP').
    descriptor_path : list of str
        List of paths to descriptor files.
    descriptor_params : dict
        Parameters specific to the chosen descriptor type.
    central_atom : str or int
        Specification of the central atom.
    xyz_path : list of str
        List of paths to XYZ coordinate files.
    xyz_base : str
        Base path for XYZ coordinate files.

    Raises
    ------
    ValueError
        If descriptor_type is None or not specified in config.
        If descriptor_path or xyz_dir are not strings or lists of strings.
        If descriptor_type is not one of 'ChEAP', 'GAPE', or 'SOAP'.
    """

    def __init__(self, config):

        self.config = config
        self.descriptor_type = config.representations.rep

        if self.descriptor_type is None:
            raise ValueError("The parameter descriptor_type has to be specified by"
                             "cfg.representations.rep or via CLI override.")

        descriptor_path_dict = {'ChEAP': config.representations.ChEAP_paths,
                                'GAPE': config.representations.GAPE_paths,
                                'SOAP': config.representations.SOAP_paths}

        descriptor_path = descriptor_path_dict[self.descriptor_type]
        if isinstance(descriptor_path, str):
            self.descriptor_path = [descriptor_path]
        elif OmegaConf.is_list(descriptor_path):
            self.descriptor_path = list(descriptor_path)
        else:
            raise ValueError("Paths should be a string or a list of strings.")

        descriptor_params_dict = {'ChEAP': config.representations.ChEAP_params,
                                  'GAPE': config.representations.GAPE_params,
                                  'SOAP': config.representations.SOAP_params}

        self.descriptor_params = descriptor_params_dict[self.descriptor_type]

        self.central_atom = config.central_atom

        if isinstance(config.xyz_dir, str):
            self.xyz_path = [config.xyz_dir]
        elif OmegaConf.is_list(config.xyz_dir):
            self.xyz_path = list(config.xyz_dir)
        else:
            raise ValueError("Paths should be a string or a list of strings.")

        self.xyz_base = config.xyz_base
        self.partitioned = config.partitioned
