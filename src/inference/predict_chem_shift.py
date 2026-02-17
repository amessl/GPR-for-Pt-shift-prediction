import joblib
import os
from hydra import initialize, compose
from generate_descriptors import GenDescriptors
import argparse
import numpy as np


def single_inference():

    with initialize(config_path="../../conf", version_base="1.1"):
        cfg = compose(config_name="config")


    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to XYZ file")
    parser.add_argument("--rep", type=str, required=True,
                        help="Name of representation to use (ChEAP, GAPE or SOAP)")


    args = parser.parse_args()

    model_path = os.path.join(cfg.backend.model.retrained_models_path,
                              f'GPR_{args.rep}.joblib')

    cfg.representations.rep = args.rep
    gen = GenDescriptors(config=cfg)

    if args.rep == 'SOAP':
        x = gen.generate_SOAP_single(input_xyz=args.input)
    elif args.rep == 'GAPE':
        x = gen.get_APE_RF_single(input_xyz=args.input)
    elif args.rep == 'ChEAP':
        x = gen.get_SIF_single(input_xyz=args.input, target_list=cfg.representations.ChEAP_params)
    else:
        raise ValueError(f'Representation {args.rep} not supported. Choose on of the following:'
                         f'"SOAP", "GAPE" or "ChEAP".')

    if os.path.exists:
        model = joblib.load(model_path)

        mean, std = model.predict(x, return_std=True)

        print(f'Chemical Shift Prediction and Uncertainty (ppm): {np.round(mean[0], 2)} +/- {np.round(std[0], 2)}')

    else:
        raise FileNotFoundError(f'Retrained model: {model_path} does not exist. '
                                f'Retrain and save model to this path.')


if __name__=='__main__':
    single_inference()
