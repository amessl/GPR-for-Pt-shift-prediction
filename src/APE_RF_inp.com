N_PROCS=4                                                                                                                 
HYPERPARAMS
2.0 3.0
500 1000 1500 2000
1e1 1e2 1e3
1 2
PATHS
'/home/alex/Pt_NMR/data/representations/APE_RF/train/' '/home/alex/Pt_NMR/data/representations/APE_RF/test/' '/home/alex/Pt_NMR/data/structures/train_split/' '/home/alex/Pt_NMR/data/structures/test_split/' '/home/alex/Pt_NMR/data/labels/train_test_split/indexed_targets_train.csv' '/home/alex/Pt_NMR/data/labels/train_test_split/indexed_targets_test.csv'
END
