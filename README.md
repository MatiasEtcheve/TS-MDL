# Multivariate Temporal Dictionary Learning for EEG

This notebook is the work produced by Omar Ahmad and Matias Etcheverry, in the course *Apprentissage pour les séries temporelles* given by Laurent Oudre and Charles Truong at the MVA.

The goal is to study the notion introduced by the article [Multivariate Temporal Dictionary Learning for EEG](https://arxiv.org/abs/1303.0742). The aim of this article is to generalize the concept of dictionary learning to multivariate signals with shift-invariant atoms.

We proposed to study this algorithm implemented on this [repository](https://github.com/sylvchev/mdla). We focus on the dataset 2a of the Brain Computer Interface IV competition (available [here](https://www.bbci.de/competition/iv/) and in the `data` folder).

**Table of content:**
1. Dictionary Learning
2. Dictionary learning in a noised setting
3. Dictionary learning with different initializations
4. Classification from activations