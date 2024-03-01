# Binarized Neural Network Metaplasticity

## Project

This repository will hold the code pertaining to Binarized Neural Network. Several papers will be implemented with several datasets available with a will to gather more information about BNN & metaplasticity within neural networks.

The main purpose of this repository is to come up with a binary metaplastic algorithm relying on Bayesian learning.

## Glossary

### Metaplasticity

Term: The ability of the synapses to change their own plasticity. In other words, the ability of a synapse
to know how important it is the the overall neural network, and to be kept or discarded depending on this
importance. Refered in [1, 2]

### BOP (Binary Optimizer)

Optimizer: One way to train binarized neural network with momentum. Even if interesting from a computational point of view, seems limited for a metaplastic implementation.

### BiNN Bayes

Optimizer: One way to train binarized bayesian neural network using the Gumbel-softmax trick.  Refered in [4].

### MESU

Optimizer: Based on [5, 6], this approach aims to bridge the gap between bayesian learning and synaptic plasticity.

## References

[1] Synaptic metaplasticity in binarized neural
networks, Axel Laborieux et al., https://www.nature.com/articles/s41467-021-22768-y.pdf

[2] Synaptic Metaplasticity in Binarized Neural Networks (GitHub Repository), Axel Laborieux, https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN

[3] Koen Helwegen, James Widdicombe, Lukas Geiger, Zechun Liu, Kwang-Ting Cheng, and Roeland Nusselder.
Latent weights do not exist: Rethinking binarized neural network optimization.

[4] Meng, Xiangming and Bachmann et al., Training Binary Neural Networks using the Bayesian Learning Rule

[5] Aitchison et al., Synaptic plasticity as Bayesian inference

[6] Blundell et al., Weight Uncertainty in Neural Networks
