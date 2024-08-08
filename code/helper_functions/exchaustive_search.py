def exhaustive_search(rcut_list, nmax_list, lmax_list, alpha_list):

    SOAP_directory = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/SOAPs/'

    XYZ_directory = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/xyz_files_final_set/'
    XYZ_base = 'st_'

    target_name = 'Experimental'
    target_path = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/Pt_II_complexes_final'


    mae_list = []
    parameter_combination_list = []

    for rcut in rcut_list:
        for nmax in nmax_list:
            for lmax in lmax_list:
                    for alpha in alpha_list:


                        try:

                            SOAP_ML = SOAP_GPR(SOAP_parameters=[rcut, nmax, lmax], SOAP_directory=SOAP_directory, XYZ_directory=XYZ_directory,
                            XYZ_base=XYZ_base, central_atom='Pt')

                            errors_std = SOAP_ML.predict(mode='read', regressor='GPR', kernel_degree=5, target_path=target_path,
                                         target_name=target_name, alpha=alpha, normalization=False)

                            mae_list.append(errors_std[0])
                            parameter_combination_list.append([rcut, nmax, lmax, alpha])

                            np.savetxt(f'/home/alex/ML/SOAP_GPR_NMR/final_dataset/prediction_errors/polynomial_kernel_unnormalized/rcut{int(rcut)}_nmax{nmax}_lmax{lmax}_alpha{alpha}_degree5_unnormalized.txt', np.array(errors_std), delimiter=',')

                        except Exception as e:
                            print(e)
                            pass

    mae_parameters_combined = [sublist + [item] for sublist, item in zip(parameter_combination_list, mae_list)]
    np.savetxt('/home/alex/ML/SOAP_GPR_NMR/final_dataset/prediction_errors/polynomial_kernel_unnormalized/mae_parameters_combined_unnormalized_degree5.txt', np.array(mae_parameters_combined), delimiter=',')

    best_params, min_error = min(zip(parameter_combination_list, mae_list), key=lambda x: x[1])

    print(best_params, min_error)

    return best_params, min_error
