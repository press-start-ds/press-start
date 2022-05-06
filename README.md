# Press Start

<p align="center">
    <img src="docs/source/img/press-start-logo.png" width="250">
</p>

## Overview

Press Start aims at providing a starting point for Data Science projects, by
running initial tests and preliminary experiments to generate insights
regarding the problem.

## How to use

Press Start uses Kedro as central framework, which allows us to execute the
steps in the project selectively.

In order to use `press-start` for with an examplet, clone this repository:

```bash
git clone git@github.com:press-start-ds/press-start.git
```
Then, enter in the folder, install the requirements and run kedro:
```bash
cd press-start
pip install -r src/requirements.txt
kedro run
```

The command should run the default pipeline and store the generated files in
the folder `data`. Notice the reports stored in `data/08_reporting`.

```bash
data
├── 01_raw
│   ├── breast_cancer_wisconsin_diagnostic.csv
│   ├── car.csv
│   └── iris.csv
├── 02_intermediate
│   ├── dev_dataset.csv
│   ├── holdout_dataset.csv
│   └── numerical_dataset.csv
├── 03_primary
│   └── feat_selection_k_best
│       ├── features.csv
│       └── metrics.csv
├── 04_feature
├── 05_model_input
├── 06_models
│   └── data_transformation
│       └── category_encoder.pkl
├── 07_model_output
└── 08_reporting
    ├── feat_selection_confusion_matrices.pdf
    └── pandas_profile.html
```


## Using `press-start` for your own projects

You change the location of the input files by editing the YAML `conf/base/catalog.yml`.

You can also change the input parameters to meet your necessities. They are defined
by the file `conf/base/parameters.yml`.

## Current pipeline

![](docs/source/img/press-start-pipeline.svg)

## Development

This project is in constant development. Keep track of the changes and feel free to contribute.
