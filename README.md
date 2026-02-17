# Uncertainty-aware prediction of 195Pt chemical shifts

![TOC Graphic](TOC_1.png)

This repository provides the datasets and machine learning models used in our publication:

> **A. Meßler and H. Bahmann*, “Uncertainty-aware prediction of 195Pt chemical shifts,” *submitted to: Journal of Chemical Information and Modeling*, 2025.**

The code and default configurations of each ML model as well as the dataset used in our work are included.

To reproduce the results for each of the three models presented in the paper, we provide a [notebook](src/example_use.ipynb). The default configs as provided in [/conf](/conf) correspond to the best models as presented in the publication.
All molecular structures used in this work can be found in [data/structures/total](data/structures/total) and the corresponding labels in a [csv-file](data/labels/total_set_clean_120525.csv). 

# Repository Strucutre

```text
.
├── conf
│   ├── backend
│   │   ├── sklearn_benchmark.yaml
│   │   └── sklearn.yaml
│   ├── config.yaml
│   ├── grid_search
│   │   └── default.yaml
│   ├── representations
│   │   ├── benchmark.yaml
│   │   └── default.yaml
│   └── splitting
│       └── default.yaml
├── data
│   ├── atomic_props.json
│   ├── benchmark
│   │   ├── labels
│   │   └── structures
│   ├── labels
│   │   ├── total_set_clean_120525.csv
│   │   └── train_test_split
│   └── structures
│       ├── test_split
│       ├── total
│       └── train_split
├── .gitignore
├── README.md
├── requirements.txt
├── src
│   ├── inference
│       ├── infer_single.py
│       └── predict_chem_shift.py
│   ├── base.py
│   ├── data_loader.py
│   ├── example_use.ipynb
│   ├── generate_descriptors.py
│   ├── get_atomic_props.py
│   ├── grid_search_CV.py
│   ├── __init__.py
│   ├── main.py
│   ├── predict_sklearn.py
│   └── stratified_split.py
└── TOC_1.png
```

## Installation

Clone the repository

```bash
git clone https://github.com/amessl/GPR-for-Pt-shift-prediction.git
cd GPR-for-Pt-shift-prediction
```
Create a python environment and use pip to install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
To directly reproduce the results provided in the publication, execute the cells for each model in the notebook [example_use.ipynb](/src/example_use.ipynb).

## Citation
If you use our models in your research, please cite:

```bibtex
@article{messler_2026_pt_nmr,
  title={Uncertainty-aware prediction of $^{195}$Pt chemical shifts},
  author={Meßler, A. and Bahmann, H.},
  journal={Journal of Chemical Information and Modeling},
  year={2026}
```
