import grid_search_CV as grid

def main(input_data):
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

    model = str(input('Representation ("SOAP" or "APE_RF"): '))

    if model == 'APE_RF':
        grid.tune_APE_RF_hyperparams(*hyperparams, paths[0], n_procs=n_procs)

    elif model == 'SOAP':
        grid.tune_SOAP_hyperparams(*hyperparams, paths[0], n_procs=n_procs)

    else:
        raise ValueError(f"Invalid model option: {model}. Choose APE_RF or SOAP.")

if __name__ == '__main__':


    main(input('Input file: '))
