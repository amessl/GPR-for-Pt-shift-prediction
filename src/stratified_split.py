#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.stats import ks_2samp
from astropy.visualization import hist
import hydra
from omegaconf import DictConfig

def stratified_split(target_data, xyz_path, split_target_path, split_xyz_dir,
                     k_bins, test_size, save_split=True, plot_dist=True):

    """

    Args:
        target_data:
        xyz_path:
        split_target_path:
        split_xyz_dir:
        save_split:
        plot_dist:

    Returns:

    """

    target_data_df = pd.read_csv(target_data)

    targets = target_data_df['Experimental']
    target_index_labels = target_data_df['Index']
    compound_names = target_data_df['Name']

    target_data_df['target_bin'] = pd.qcut(targets, q=k_bins, labels=False)
    pseudo_cluster_labels = target_data_df['target_bin']


    index_train, index_test, \
        y_train, y_test = train_test_split(target_index_labels, targets,
                                        stratify=pseudo_cluster_labels, random_state=42,
                                        test_size=test_size)

    names_train, names_test, \
        y_train, y_test = train_test_split(compound_names, targets,
                                        stratify=pseudo_cluster_labels, random_state=42,
                                        test_size=test_size)


    xyz_train_path = os.path.join(split_xyz_dir, 'train_split')
    xyz_test_path = os.path.join(split_xyz_dir, 'test_split')

    if os.path.exists(xyz_train_path):
        shutil.rmtree(xyz_train_path)

    if os.path.exists(xyz_test_path):
        shutil.rmtree(xyz_test_path)

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

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    index_train, index_test, \
        y_train, y_test = stratified_split(target_data=cfg.splitting.target.original_target,
                                        xyz_path=cfg.splitting.structures.original_xyz,
                                        split_target_path=cfg.splitting.target.split_target,
                                        split_xyz_dir=cfg.splitting.structures.split_xyz,
                                        k_bins=cfg.splitting.splitting.k_quantiles,
                                        test_size=cfg.splitting.splitting.test_size,
                                        save_split=True)

    if cfg.splitting.splitting.run_ks:
        test_target_dist(y_train, y_test)


if __name__ == '__main__':
    main()
