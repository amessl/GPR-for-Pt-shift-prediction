# Create learning curves for r_cut dependency

import matplotlib.pyplot as plt

rcut_list = [2.0, 3.0, 4.0, 5.0, 6.0]
mean_MAE_rcut_list = []

for rcut in rcut_list:
    SOAP_ML = SOAP_GPR(SOAP_parameters=[rcut, 8, 8], SOAP_directory=SOAP_directory, XYZ_directory=XYZ_directory,
                            XYZ_base='st_', central_atom='Pt')

    errors_std = SOAP_ML.predict(mode='read', regressor='GPR', kernel_degree=1, target_path=target_path,
                target_name=target_name, alpha=10.0, normalization=False, lc=True)

    print(errors_std)

    mean_MAE_list = []

    lc_MAE = errors_std[5]
    train_sizes = errors_std[4]

    for row in lc_MAE:
        mean_MAE = np.mean(np.abs(row))
        mean_MAE_list.append(mean_MAE)

    print(mean_MAE_list)
    print(train_sizes)
    mean_MAE_rcut_list.append(mean_MAE_list)


color_list = ['orange', 'r', 'purple', 'b', 'k']
#color_list = ['k', 'b', 'purple', 'r', 'orange']

for i, rcut in enumerate(rcut_list):
    color = color_list[i] # Use colormap to define color

    # Extract the mean MAE values and train sizes for the current rcut
    mean_MAE_list_rcut = mean_MAE_rcut_list[i]
    train_sizes_rcut = train_sizes

    # Plot the curve
    plt.plot(train_sizes_rcut, mean_MAE_list_rcut, color=color)
    plt.scatter(train_sizes_rcut, mean_MAE_list_rcut, color=color, label=f'r={rcut}')

# Add labels and legend
plt.xlabel('Number of samples in training set')
plt.ylabel('Mean Absolute Error [ppm]')
plt.legend()
plt.grid()

plt.savefig(f'/home/alex/ML/SOAP_GPR_NMR/final_dataset/figures/degree_dependency/lc_rcut{rcut}_n{8}_l{8}_unnormalized_final.svg', format='svg', dpi=700, bbox_inches='tight')#
plt.show()
