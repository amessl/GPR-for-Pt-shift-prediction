# How to create heatmap of n_max, l_max
import seaborn as sns
import numpy as np

nl_matrix = np.zeros((8, 8))

for n_max in range(1,9):
    for l_max in range(1,9):

        SOAP_ML = SOAP_GPR(SOAP_parameters=[6.0, n_max, l_max], SOAP_directory=SOAP_directory,
                           XYZ_directory=XYZ_directory, XYZ_base='st_', central_atom='Pt')

        nl_matrix[n_max-1, l_max-1] = SOAP_ML.predict(mode='read', regressor='GPR', kernel_degree=2, target_path=target_path,
             target_name=target_name, alpha=1e-3, normalization=True)[0]


sns.heatmap(nl_matrix, cmap="hot", annot=True, fmt=".1f",
            xticklabels=['1','2','3','4','5','6','7','8'], yticklabels=['1','2','3','4','5','6','7','8'])
