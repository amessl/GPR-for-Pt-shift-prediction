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
from astropy.visualization import hist


def get_clusters(descriptor_path, xyz_path, eps, min_samples,
                 save_clusters=True, target_path=None, red_dim=True,
                 n_comp=None, plot_clusters=True):

    gen = generate_descriptors(descriptor_params=['EN', 'alpha', 'val', 'qmol'],
                               descriptor_path=descriptor_path, central_atom='Pt', xyz_path=xyz_path,
                               xyz_base='st_')
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

    labels_occuring_once = unique_labels[counts == 1]

    if labels_occuring_once is not None:
        print(f'Cluster labels that occur only once: {labels_occuring_once}')
        print('\n')

    else:
        print('No lonely clusters detected.')
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

        print(f'Target data together with cluster labels saved to: {target_path}_dbscan_pca.csv')
        print('\n')

    return clusters, n_clusters


def stratified_split(target_path, xyz_path, split_target_path, split_xyz_dir, save_split=True, plot_dist=True):
    total_data_cluster_labels = pd.read_csv(f'{target_path}.csv')

    target_data = total_data_cluster_labels['Experimental']
    cluster_labels = total_data_cluster_labels['dbscan_pca_1']
    target_index_labels = total_data_cluster_labels['Index']

    index_train, index_test, \
        y_train, y_test = train_test_split(target_index_labels, target_data,
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

        indexed_targets_train = np.column_stack((index_train, y_train))
        indexed_targets_train_df = pd.DataFrame(indexed_targets_train, columns=["Index", "Experimental"])
        indexed_targets_train_df.to_csv(os.path.join(split_target_path, 'indexed_targets_train.csv'), index=False)

        indexed_targets_test = np.column_stack((index_test, y_test))
        indexed_targets_test_df = pd.DataFrame(indexed_targets_test, columns=["Index", "Experimental"])
        indexed_targets_test_df.to_csv(os.path.join(split_target_path, 'indexed_targets_test.csv'), index=False)

    if plot_dist:

        hist(y_train, bins='freedman', density=False)
        plt.show()

        hist(y_test, bins='freedman', density=False)
        plt.show()


    else:
        print('Indexed Train/Test split not saved. Set "save_split=True" to save splits.')

    return


SIF_dir = '/home/alex/Pt_NMR/data/representations/SIF/'
XYZ_dir = '/home/alex/Pt_NMR/data/structures/total/'

target_path_original = '/home/alex/Pt_NMR/data/labels/final_data_corrected'
target_name = 'Experimental'

#get_clusters(descriptor_path=SIF_dir, xyz_path=XYZ_dir, eps=0.5, min_samples=5,
#             save_clusters=True, target_path=target_path_original, red_dim=True,
#             n_comp=2, plot_clusters=True)


target_path_indexed = '/home/alex/Pt_NMR/data/labels/final_data_corrected_dbscan_pca'

split_target_path = '/home/alex/Pt_NMR/data/labels/train_test_split/'



stratified_split(target_path_indexed, xyz_path=XYZ_dir, split_target_path=split_target_path,
                 split_xyz_dir='/home/alex/Pt_NMR/data/structures/', save_split=True)




