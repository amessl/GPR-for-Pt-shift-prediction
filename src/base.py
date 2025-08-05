from omegaconf import OmegaConf


class BaseConfig:

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
