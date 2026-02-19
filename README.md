# Uncertainty-aware prediction of 195Pt chemical shifts

![TOC Graphic](TOC_1.png)

This repository provides the datasets and machine learning models used in our publication:

> **A. Meßler and H. Bahmann*, “Uncertainty-aware prediction of 195Pt chemical shifts from limited data” *published in Journal of Chemical Information and Modeling*, 2026.**

The code and default configurations of each ML model as well as the dataset used in our work are included.

To reproduce the results for each of the three models presented in the paper, we provide a [notebook](src/example_use.ipynb). The default configs as provided in [/conf](/conf) correspond to the best models as presented in the publication.
All molecular structures used in this work can be found in [data/structures/total](data/structures/total) and the corresponding labels in a [csv-file](data/labels/total_set_clean_120525.csv). 

# Repository Strucutre

```text
.
├── app
│   ├── streamlit_app.py
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
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
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

### Docker image
To use the final models with a UI for inference only, obtain the docker image from the dockerhub repository and run the container locally. Installing python and various dependencies is avoided this way. Install docker on your system and then run the following command:
```bash
docker pull amessl/gpr-for-pt-shift-prediction:1.0.1
```

### Build from source
To use the final models for inference and access the whole codebase, do the following:

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

Adjust the data_root in [config.yaml](/conf/config.yaml) to match the path of your local environment.

## Usage
To directly reproduce the results provided in the publication, execute the cells for each model in the notebook [example_use.ipynb](/src/example_use.ipynb).
### 195Pt chemical shift prediction via CLI
Inference on new structures with the final model (retrained on the total dataset using one of the three representations) can be carried out by executing the prediction pipeline:
```bash
python -m src.inference.infer_single --input [path to xyz-file] --rep [representation (ChEAP, GAPE or SOAP)]
```
The output consists of the chemical shit prediction and the corresponding prediction uncertainties.
Please make sure that your input for the CLI tool is a xyz-file containing Cartesian atomic coordinates with the example format:
```bash
7 # total number of atoms
-2 # charge of the Pt complex
Pt  1.00476781399991      0.10050017558456     -0.00902329070933
Br  1.00385941522517      0.82895807628853      2.40742215132927
Cl  1.00457090376522      2.36597132620999     -0.69145643454121
Cl  -1.36120352054679      0.09999982105003     -0.00973068784916
Br  1.00565552307436     -0.62794205666902     -2.42546262306348
Cl  1.00497122637463     -2.16495216376531      0.67340550015585
Cl  3.37072863810750      0.10098482130122     -0.00831461532193
```
Currently the models are built to work for mononuclear Pt complexes only.

### 195Pt chemical shift prediction via UI
The same can be done when using the docker image. After pulling the image from the repository as explained above, run the following commands:

```bash
docker run -p 8501:8501 amessl/gpr-for-pt-shift-prediction
```

Access the UI via the local URL:

```bash
http://localhost:8501
```

## Citation
If you use our models in your research, please cite:

```bibtex
@article{doi:10.1021/acs.jcim.5c02541,
author = {Meßler, Alexander and Bahmann, Hilke},
title = {Uncertainty-Aware Prediction of 195Pt Chemical Shifts from Limited Data},
journal = {Journal of Chemical Information and Modeling},
volume = {66},
number = {3},
pages = {1498-1510},
year = {2026},
doi = {10.1021/acs.jcim.5c02541},
    note ={PMID: 41610407},
URL = {https://doi.org/10.1021/acs.jcim.5c02541}
}
```
