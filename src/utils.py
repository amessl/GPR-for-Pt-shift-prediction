import numpy as np
from hydra import initialize, compose
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from data_loader import DataLoader
from predict_sklearn import SklearnGPRegressor
from IPython.display import display

# Out-of-pipeline usage of descriptor generation class for plots and data analysis

def inspect_cheap(props):
    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name="config")

    cfg.representations.rep = 'ChEAP'
    cfg.representations.rescale_ChEAP = False
    cfg.representations.ChEAP_params = props

    loader = DataLoader(config=cfg)

    shift_vals = []
    samples = []
    compound_names = []

    try:
        shift_vals = loader.load_targets(partitioned=False)[0]
        samples = loader.load_samples(partitioned=False)[0]

        compound_names = loader.load_targets(partitioned=False)[4]
    except Exception as e:
        print(e)


    print(len(shift_vals))

    return shift_vals, samples, compound_names


def plot_cheap(props, xlabel, save=False, annotate=False):
    mpl.rcParams.update(mpl.rcParamsDefault)  # or: plt.rcdefaults()
    plt.style.use('default')

    shift_vals, samples, names = inspect_cheap(props)

    valency_list = []
    prop_list = []

    for sample in samples:
        prop_list.append(sample[0])
        valency_list.append(sample[1])

    # Assign valency value of 4 to complexes with ethene ligands for consistency of geometry
    valency_list_corr = list(map(lambda x: 4 if x == 5 else x, valency_list))

    colors = ['red', 'b']
    #labels = set(valency_list_corr)
    labels = ['Square-planar', 'Octahedral']
    plot = plt.scatter(prop_list, shift_vals, c=valency_list_corr, cmap=mpl.colors.ListedColormap(colors))
    plt.legend(handles=plot.legend_elements()[0], labels=labels, title='Complex Geometry', fontsize=12, title_fontsize='large')

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Chemical shift (ppm)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid()

    if annotate:
        plt.annotate('$R=0.94$', xy=(0.15, 0.3), xycoords='axes fraction',  # 0.94/0.99 for EN, -0.6/-0.94 for Z, -0.73/-0.99 for r
                     fontsize=14, color='red', ha='center')

        plt.annotate('$R=0.99$', xy=(0.5, 0.65), xycoords='axes fraction',
                     fontsize=14, color='blue', ha='center')

    if save:
        plt.savefig(f'../paper/figs/scatter{props[0]}_annotated.png', dpi=400, bbox_inches='tight')


    df = pd.DataFrame({'Shift': shift_vals,
                       'Name': names,
                       'Mean Prop': prop_list,
                       'Valency': valency_list})

    display(df)

    df.to_csv('../paper/figs/data_for_figs/mean_props_for_scatter_plot.csv')

    plt.show()


def plot_cheap_qmol(props, save=False, x_label='Mean Electronegativity'):
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('default')

    shift_vals, samples = inspect_cheap(props)

    qmol_list = []
    prop_list = []

    for sample in samples:
        prop_list.append(sample[0])
        qmol_list.append(sample[1])

    dot_labels = sorted(set(qmol_list))
    colors = ['red', 'green', 'blue', 'k', 'y']

    plot = plt.scatter(prop_list, shift_vals, c=qmol_list, cmap=mpl.colors.ListedColormap(colors))
    plt.legend(handles=plot.legend_elements()[0], labels=dot_labels, title='Molecular Charge', fontsize=14)

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Chemical shift (ppm)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid()

    if save:
        plt.savefig('../paper/figs/fig_3.png', dpi=400, bbox_inches='tight')

    plt.show()


def get_errors_barplot(rep, noise):
    """
    Plotting error overview for SIF
    """

    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name="config")

    cfg.representations.rep = rep

    model = SklearnGPRegressor(config=cfg)

    degrees = [1, 2, 3, 4, 5]

    # GAPE-specific params
    rcuts_gape = [4.0, 3.0, 2.0, 2.0, 4.5]
    dims = [2500, 500, 500, 500, 500]

    # SOAP-specific params
    rcuts_soap = [2.0, 2.0, 2.0, 2.0, 2.0]
    nmax_list = [3, 3, 2, 1, 1]
    lmax_list = [6, 5, 5, 6, 6]

    train_mae = []
    train_rmse = []

    test_mae = []
    test_rmse = []

    if rep == 'ChEAP':
        for degree in degrees:
            train_errors = model.gpr_train(kernel_degree=degree, noise=noise)

            test_errors = model.gpr_test(kernel_degree=degree, noise=noise)

            train_mae.append(train_errors[0])
            train_rmse.append(train_errors[2])

            test_mae.append(test_errors[0])
            test_rmse.append(test_errors[1])

    elif rep == 'GAPE':
        for degree, rcut, dim in zip(degrees, rcuts_gape, dims):

            cfg.representations.GAPE_params['rcut'] = rcut
            cfg.representations.GAPE_params['dim'] = dim

            train_errors = model.gpr_train(kernel_degree=degree, noise=noise)

            test_errors = model.gpr_test(kernel_degree=degree, noise=noise)

            train_mae.append(train_errors[0])
            train_rmse.append(train_errors[2])

            test_mae.append(test_errors[0])
            test_rmse.append(test_errors[1])

    elif rep == 'SOAP':
        for degree, rcut, nmax, lmax in zip(degrees, rcuts_soap, nmax_list, lmax_list):

            cfg.representations.SOAP_params['rcut'] = rcut
            cfg.representations.SOAP_params['nmax'] = nmax
            cfg.representations.SOAP_params['lmax'] = lmax

            train_errors = model.gpr_train(kernel_degree=degree, noise=noise)

            test_errors = model.gpr_test(kernel_degree=degree, noise=noise)

            train_mae.append(train_errors[0])
            train_rmse.append(train_errors[2])

            test_mae.append(test_errors[0])
            test_rmse.append(test_errors[1])

    # results_SIF = np.row_stack([train_mae, train_rmse, test_mae, test_rmse])
    # np.savetxt('/home/alex/Pt_NMR/paper/figs/data_for_figs/results_SIF.txt', results_SIF)

    return train_mae, test_mae, train_rmse, test_rmse


def boxplot(bp_type: str, mark_out=True, retrain=False):

    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name="config")

    reps = ['ChEAP', 'GAPE', 'SOAP']
    bp_list = []

    for rep in reps:
        cfg.representations.rep = rep
        model = SklearnGPRegressor(config=cfg)

        if retrain:
            model.gpr_train(kernel_degree=2, noise=0.01)

        if bp_type == 'Residuals':
            gpr_test_output = model.gpr_test(kernel_degree=2, noise=0.01)
            bp_data = gpr_test_output[4]
            names = gpr_test_output[5]


            zipped = zip(list(bp_data), names)

            print('-' * 35)
            print(f'Model: {rep}')
            print('-'*35)
            print(f'Sorted outliers (Values and Compound names: {sorted(zipped, key=lambda x: x[0])}')

            ylabel = 'Prediction Error (ppm)'

            # Compute Q1, Q3, and IQR
            #Q1 = np.percentile(bp_data, 25)
            #Q3 = np.percentile(bp_data, 75)
            #IQR = Q3 - Q1

            # Define outlier bounds
            #lower_bound = Q1 - 1.5 * IQR
            #upper_bound = Q3 + 1.5 * IQR

            #outliers = bp_data[(bp_data < lower_bound) | (bp_data > upper_bound)]
            #print(f'Outlier test results: {outliers}')

            bp_list.append(bp_data)

        elif bp_type == 'Uncertainty':
            gpr_test_output = model.gpr_test(kernel_degree=2, noise=0.01)

            bp_data = gpr_test_output[3]
            names = gpr_test_output[5]

            zipped = zip(list(bp_data), names)
            print('-' * 35)
            print(f'Model: {rep}')
            print('-'*35)
            print(f'Sorted outliers (Values and Compound names: {sorted(zipped, key=lambda x: x[0])}')

            ylabel = 'Uncertainty (ppm)'

            # Compute Q1, Q3, and IQR
            #Q1 = np.percentile(bp_data, 25)
            #Q3 = np.percentile(bp_data, 75)
            #IQR = Q3 - Q1

            # Define outlier bounds
            #lower_bound = Q1 - 1.5 * IQR
            #upper_bound = Q3 + 1.5 * IQR

            #outliers = bp_data[(bp_data < lower_bound) | (bp_data > upper_bound)]
            #print(f'Outlier test results: {outliers}')

            bp_list.append(bp_data)

    ChEAP_data = bp_list[0]
    GAPE_data = bp_list[1]
    SOAP_data = bp_list[2]

    # Example DataFrame
    df = pd.DataFrame({
        'SOAP': SOAP_data,
        'GAPE': GAPE_data,
        'ChEAP': ChEAP_data
    })

    df_melted = df.melt(var_name='Model', value_name=str(bp_type))

    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=df_melted, x='Model', y=str(bp_type))

    if mark_out:

        for i, model in enumerate(df.columns):
            values = df[model].dropna()
            stats = boxplot_stats(values, whis=1.5)[0]
            outliers = stats['fliers']
            print(f'Outliers: {outliers}')
            if len(outliers) > 0:
                max_outlier = min(outliers)
                ax.scatter(i, max_outlier, color="red", zorder=10, s=50,
                           label="$trans-[Pt(SMe_{2})I_{2}]$" if i == 0 else "")

    plt.xlabel('')
    plt.ylabel('Error (ppm)', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=16)

    # Only show legend once
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels, fontsize=14)

    plt.savefig(f'../paper/figs/bp_total_{bp_type}_new.png', dpi=400, bbox_inches='tight')
    plt.show()


