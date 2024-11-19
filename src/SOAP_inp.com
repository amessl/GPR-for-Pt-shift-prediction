N_PROCS=4
HYPERPARAMS 
2.0 3.0
1 2 3
1 2 3
1e-1 1e0 1e1
2 3
PATHS
'/home/alex/Pt_NMR/data/representations/SOAP/train/' '/home/alex/Pt_NMR/data/representations/SOAP/test/' '/home/alex/Pt_NMR/data/structures/train_split/' '/home/alex/Pt_NMR/data/structures/test_split/' '/home/alex/Pt_NMR/data/labels/train_test_split/indexed_targets_train.csv' '/home/alex/Pt_NMR/data/labels/train_test_split/indexed_targets_test.csv' 
END
