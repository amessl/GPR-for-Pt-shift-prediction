#!/usr/bin/env python3

import grid_search_CV as grid
import argparse


def main(input_data, representation):
    hyperparams = []
    paths = []
    n_procs = 1
    section = None

    with open(input_data, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('N_PROCS'):
                n_procs = int(line.split('=')[1])
            elif line == 'HYPERPARAMS':
                section = 'hyperparams'
                continue
            elif line == 'PATHS':
                section = 'paths'
                continue
            elif line == 'END':
                break

            if section == 'hyperparams':
                hyperparams.append([float(x) if '.' in x or 'e' in x else int(x) for x in line.split()])

            elif section == 'paths':
                paths.append([x.strip("'") for x in line.split()])

    if representation == 'APE_RF':
        grid.tune_APE_RF_hyperparams(*hyperparams, paths[0], n_procs=n_procs)

    elif representation == 'SOAP':
        grid.tune_SOAP_hyperparams(*hyperparams, paths[0], n_procs=n_procs)

    else:
        raise ValueError(f"Invalid representation option: {representation}. Choose APE_RF or SOAP.")

    return hyperparams, paths, n_procs

if __name__ == '__main__':
    parsing = argparse.ArgumentParser(description='Run grid search for hyperparameter optimization')

    parsing.add_argument('--input', '-i', type=str, help='Provide the path to the input file '
                                                   'for the hyperarameter optimization', required=True)

    parsing.add_argument('--rep', '-r', type=str, help='Provide the name of the representation '
                                                 'you want to use (SOAP or APE_RF)', required=True)
    args = parsing.parse_args()

    main(args.input, args.rep)
