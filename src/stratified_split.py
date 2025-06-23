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

def stratified_split(target_data, xyz_dir, split_target_path, split_xyz_dir,
                     k_bins, test_size, save_split=True, plot_dist=True):
    """
    Perform a stratified train/test split on experimental target data by binning into quantile-based clusters.
    Also splits associated .xyz files and optionally saves and plots distributions.

    Args:
        target_data (str): Path to a CSV file containing the full dataset with at least 'Experimental', 'Index', and 'Name' columns.
        xyz_dir (str): Directory where the original set of .xyz files is stored.
        split_target_path (str): Directory where the resulting train/test label CSVs will be saved.
        split_xyz_dir (str): Directory where the train/test-split of .xyz samples will be stored.
        k_bins (int): Number of quantile bins to use for stratification.
        test_size (float): Proportion of the dataset to include in the test split (between 0 and 1).
        save_split (bool, optional): If True, saves the split label files to `split_target_path`. Defaults to True.
        plot_dist (bool, optional): If True, plots histograms of the train and test target distributions. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - index_train (pd.Series): Indices of training samples.
            - index_test (pd.Series): Indices of test samples.
            - y_train (pd.Series): Experimental values for training samples.
            - y_test (pd.Series): Experimental values for test samples.
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
        xyz_file = os.path.join(xyz_dir, f'st_{index}.xyz')
        xyz_train_file = os.path.join(xyz_train_path, f'st_{index}.xyz')

        shutil.copy2(xyz_file, xyz_train_file)

    for index in index_test:
        xyz_file = os.path.join(xyz_dir, f'st_{index}.xyz')
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
    """
    Perform a two-sample Kolmogorov-Smirnov (KS) test to compare the distributions
    of training and test target values.

    Args:
        train_targets (array-like): Target values (labels) in the training set.
        test_targets (array-like): Target values (labels) in the test set.

    Returns:
        tuple: A tuple containing:
            - ks_stat (float): KS test statistic indicating the maximum difference between CDFs.
            - p_value (float): p-value for the test; a high value indicates the two distributions are similar.
    """

    print('Performing Kolmogorov-Smirnov Test to compare target distribution in stratified split.', '\n')
    ks_stat, p_value = ks_2samp(train_targets, test_targets)

    print('KS Stats:', ks_stat)
    print('p-value:', p_value)

    return ks_stat, p_value

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """
   Main entry point for performing stratified splitting and optional distribution testing.

   This function loads configuration via Hydra, performs a stratified train/test split
   on a labeled dataset of molecules and associated structure files, and optionally runs
   a Kolmogorov-Smirnov test to compare label distributions.

   Args:
       cfg (DictConfig): Hydra configuration object containing all required paths and parameters:
           - splitting.target.original_target (str): Path to original label CSV.
           - splitting.structures.original_xyz (str): Directory containing original .xyz files.
           - splitting.target.split_target (str): Output directory for split label CSVs.
           - splitting.structures.split_xyz (str): Output directory for split .xyz files.
           - splitting.splitting.k_quantiles (int): Number of quantile bins to use for stratification.
           - splitting.splitting.test_size (float): Proportion of data for test split.
           - splitting.splitting.run_ks (bool): Whether to perform KS test on the target distributions.
    """

    index_train, index_test, \
        y_train, y_test = stratified_split(target_data=cfg.splitting.target.original_target,
                                        xyz_dir=cfg.splitting.structures.original_xyz,
                                        split_target_path=cfg.splitting.target.split_target,
                                        split_xyz_dir=cfg.splitting.structures.split_xyz,
                                        k_bins=cfg.splitting.splitting.k_quantiles,
                                        test_size=cfg.splitting.splitting.test_size,
                                        save_split=True)

    if cfg.splitting.splitting.run_ks:
        test_target_dist(y_train, y_test)


if __name__ == '__main__':
    main()
