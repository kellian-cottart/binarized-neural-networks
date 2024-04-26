# Binarized Bayesian Metaplasticity

## Description

This repository holds the code pertaining to "Binarized Bayesian Metaplasticity".

To install the environment needed to run the code, please use either pip or conda. Make sure you are running either of the commands in the main directory.

Using Pip:

```bash
pip install -r requirements
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate binarized
```

## Architecture

The main file to start simulations is `./deepMain.py`.

The repository is ordered as follows:

- dataloader: folder containing all functions neccessary to download, transform and load datasets in PyTorch tensors;
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

All hyperparameters were found using Optuna [?] Hyperparameter search.

## References

[1] Synaptic metaplasticity in binarized neural
networks, Axel Laborieux et al., https://www.nature.com/articles/s41467-021-22768-y.pdf

[2] Synaptic Metaplasticity in Binarized Neural Networks (GitHub Repository), Axel Laborieux, https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN

[3] Koen Helwegen, James Widdicombe, Lukas Geiger, Zechun Liu, Kwang-Ting Cheng, and Roeland Nusselder.
Latent weights do not exist: Rethinking binarized neural network optimization.

[4] Meng, Xiangming and Bachmann et al., Training Binary Neural Networks using the Bayesian Learning Rule

[5] Aitchison et al., Synaptic plasticity as Bayesian inference

[6] Blundell et al., Weight Uncertainty in Neural Networks
