N_PROCS=4                                                                                                                 
HYPERPARAMS
2.0 2.5 3.0 3.5 4.0
500 750 1000 1250
1e1
1 2 3 4 5
PATHS
'/home/alex/Pt_NMR/data/representations/APE_RF/train/' '/home/alex/Pt_NMR/data/representations/APE_RF/test/' '/home/alex/Pt_NMR/data/structures/train_split/' '/home/alex/Pt_NMR/data/structures/test_split/' '/home/alex/Pt_NMR/data/labels/train_test_split/indexed_targets_train.csv' '/home/alex/Pt_NMR/data/labels/train_test_split/indexed_targets_test.csv'
END
