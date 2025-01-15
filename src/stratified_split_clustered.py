#!/usr/bin/env python3

from generate_descriptors import generate_descriptors
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import shutil
from scipy.stats import ks_2samp
from astropy.visualization import hist
import argparse
import json


def get_clusters(descriptor_path, xyz_path, eps, min_samples,
                 save_clusters=False, target_path=None, red_dim=True,
                 n_comp=None, plot_clusters=True):

    gen = generate_descriptors(descriptor_params=['EN', 'alpha', 'val', 'qmol'],
                               descriptor_path=descriptor_path, central_atom='Pt', xyz_path=xyz_path,
                               xyz_base='st_', normalize=False)
    X_data = gen.get_SIF()[0]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_data)

    if red_dim:
        pca = PCA(n_components=n_comp)
        data_pca = pca.fit_transform(data_scaled)

        print(f'Explained variance ratio by PCA components: {pca.explained_variance_ratio_}')
        print('\n')

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_pca)
        clusters = clustering.labels_

    else:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
        clusters = clustering.labels_

    unique_labels, counts = np.unique(clusters, return_counts=True)
    n_clusters = len(unique_labels)

    print(f'Number of clusters found with DBSCAN: {n_clusters}')
    print('\n')

    if plot_clusters:

        if n_comp == 2:

            cmap = plt.get_cmap('tab10', n_clusters)

            norm = mcolors.BoundaryNorm(boundaries=np.arange(min(clusters) - 0.5, max(clusters) + 1.5, 1),
                                        ncolors=cmap.N)

            plt.figure(figsize=(10, 7))
            plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap=cmap, norm=norm, s=100, edgecolor='k',
                        alpha=0.6)

            cbar = plt.colorbar(ticks=unique_labels)
            cbar.set_label('Cluster Labels')

            plt.title('DBSCAN Clustering of PCA-Reduced Data')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.show()

        else:
            raise ValueError('For "2D plot of clusters define "n_clusters=2"')

    else:
        pass


    if save_clusters:
        total_data = pd.read_csv(f'{target_path}.csv')

        total_data['dbscan_pca_1'] = list(clusters)
        total_data.to_csv(f'{target_path}_dbscan_pca.csv')

        print(f'Target data together with cluster labels saved to: \n {target_path}_dbscan_pca.csv')
        print('\n')

    lonely_labels = unique_labels[counts == 1]

    if lonely_labels is not None:
        print(f'Cluster labels that occur only once: {lonely_labels}')
        print('\n')

        clustered_target = pd.read_csv(f'{target_path}_dbscan_pca.csv')

        for label in lonely_labels:

            lonely_label_df = clustered_target[clustered_target['dbscan_pca_1'] == label]
            lonely_label_df.to_csv(f'lonely_cluster_{label}.csv', index=False)

            index_name = clustered_target[clustered_target['dbscan_pca_1'] == label].index


            clustered_target = clustered_target.drop(index_name[0])

            print(clustered_target[180:195])

            print(
                f'Target values corresponding to lonely clusters moved from target data to file: \n '
                f'lonely_cluster_{label}.txt', '\n')

    else:
        print('No lonely clusters detected.')
        print('\n')

    return clusters, clustered_target


def stratified_split(target_data, xyz_path, split_target_path, split_xyz_dir, save_split=False, plot_dist=True):
    total_data_cluster_labels = target_data

    #print(total_data_cluster_labels[180:195])

    target_data = total_data_cluster_labels['Experimental']
    cluster_labels = total_data_cluster_labels['dbscan_pca_1']
    target_index_labels = total_data_cluster_labels['Index']
    compound_names = total_data_cluster_labels['Name']

    index_train, index_test, \
        y_train, y_test = train_test_split(target_index_labels, target_data,
                                           stratify=cluster_labels, random_state=42, test_size=0.2)

    names_train, names_test, \
        y_train, y_test = train_test_split(compound_names, target_data,
                                           stratify=cluster_labels, random_state=42, test_size=0.2)

    xyz_train_path = os.path.join(split_xyz_dir, 'train_split')
    xyz_test_path = os.path.join(split_xyz_dir, 'test_split')

    os.makedirs(xyz_train_path, exist_ok=True)
    os.makedirs(xyz_test_path, exist_ok=True)


    for index in index_train:
        xyz_file = os.path.join(xyz_path, f'st_{index}.xyz')
        xyz_train_file = os.path.join(xyz_train_path, f'st_{index}.xyz')

        shutil.copy2(xyz_file, xyz_train_file)

    for index in index_test:
        xyz_file = os.path.join(xyz_path, f'st_{index}.xyz')
        xyz_test_file = os.path.join(xyz_test_path, f'st_{index}.xyz')

        shutil.copy2(xyz_file, xyz_test_file)

    if save_split:

        os.makedirs(split_target_path, exist_ok=True)

        indexed_targets_train = np.column_stack((index_train, y_train, names_train))
        indexed_targets_train_df = pd.DataFrame(indexed_targets_train, columns=["Index", "Experimental", "Name"])
        indexed_targets_train_df.to_csv(os.path.join(split_target_path, 'indexed_targets_train.csv'), index=False)

        indexed_targets_test = np.column_stack((index_test, y_test, names_test))
        indexed_targets_test_df = pd.DataFrame(indexed_targets_test, columns=["Index", "Experimental", "Name"])
        indexed_targets_test_df.to_csv(os.path.join(split_target_path, 'indexed_targets_test.csv'), index=False)

    if plot_dist:

        hist(y_train, bins='freedman', density=False)
        plt.show()

        hist(y_test, bins='freedman', density=False)
        plt.show()


    else:
        print('Indexed Train/Test split not saved. Set "save_split=True" to save splits.')

    return index_train, index_test, y_train, y_test


def test_target_dist(train_targets, test_targets):

    print('Performing Kolmogorov-Smirnov Test to compare target distribution in stratified split.', '\n')
    ks_stat, p_value = ks_2samp(train_targets, test_targets)

    print('KS Stats:', ks_stat)
    print('p-value:', p_value)

    return ks_stat, p_value

# TODO: Table with data hyperparameter optimization data
# TODO: Table with test error for each top candidate (table as csv)

if __name__ == '__main__':
    parsing = argparse.ArgumentParser(description='Create stratified train-test-split based on '
                                                  'cluster labels obtained using DBSCAN.')

    parsing.add_argument('--input', '-i', type=str, help='Provide path to JSON file containing required paths '
                                                         'to target data, structures and clustering parameters',
                                                                                                    required=True)

    parsing.add_argument('--pca', help='Specify if PCA is to be performed before clustering',
                         action=argparse.BooleanOptionalAction)

    parsing.add_argument('--ks', help='Specify if KS test is to be performed after stratified split.',
                         action=argparse.BooleanOptionalAction)

    args = parsing.parse_args()

    categories = ['target', 'structures', 'clustering']

    with open(args.input) as file:
        input_data = json.load(file)

    target_paths = input_data[categories[0]]
    structure_paths = input_data[categories[1]]
    cluster_params = input_data[categories[2]]

    if args.pca:

        clusters, clustered_targets = get_clusters(descriptor_path=cluster_params['clustering_features'], xyz_path=structure_paths['original_xyz'],
                                      eps=cluster_params['clustering_params'][0], min_samples=cluster_params['clustering_params'][1],
                                      save_clusters=True, target_path=target_paths['original_target'], red_dim=True,
                                      n_comp=cluster_params['clustering_params'][2], plot_clusters=True)

    else:

        clusters, clustered_targets = get_clusters(descriptor_path=cluster_params['clustering_features'], xyz_path=structure_paths['original_xyz'],
                                      eps=cluster_params['clustering_params'][0], min_samples=cluster_params['clustering_params'][1],
                                      save_clusters=True, target_path=target_paths['original_target'], red_dim=False, plot_clusters=True)



    index_train, index_test, y_train, y_test = stratified_split(target_data=clustered_targets, xyz_path=structure_paths['original_xyz'],
                                               split_target_path=target_paths['split_target'], split_xyz_dir=structure_paths['split_xyz'],
                                               save_split=True)

    if args.ks:
        test_target_dist(y_train, y_test)