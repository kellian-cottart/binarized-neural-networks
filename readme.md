# Binarized Bayesian Metaplasticity

## Description

This repository holds the code pertaining to "Binarized Bayesian Metaplasticity".

To install the environment needed to run the code, please use conda. Make sure you are running either of the commands in the main directory.

```bash
conda env create -f environment.yml
conda activate binarized-learning-env
```

## Architecture

The main file to start simulations is `./deepMain.py`.

The repository is ordered as follows:

- dataloader: folder containing all functions necessary to download, transform and load datasets in PyTorch tensors;
- datasets: folder containing the downloaded or transformed datasets;
- models: folder containing the classes defining networks, with embedded folders holding layers and specific activation functions;
- notebooks: folder containing notebooks for certain specific figures of the article;
- optimizers: folder containing all optimizers (Adam, Synaptic Metaplasticity [1]...);
- trainer: folder containing a framework class used to train the neural networks on the different dataset;
- utils: folder containing the functions necessary to plot graphs into the folder `./saved_deep_models`.

## Results reproducibility

To yield the figures in the article, please enter the following commands:

TODO

## Hyperparameter Optimization

All hyperparameters were found using Optuna Hyperparameter search.

