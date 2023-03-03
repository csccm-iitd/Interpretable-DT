# Probabilistic-predictive-and-interpretable-digital-twin
Probabilistic machine learning based predictive and interpretable digital twin for dynamical systems

# Learning-physics-from-output-only-data
This repository contains the python codes of the paper 
  > + Tripura, T., Desai, A. S., Adhikari, S., & Chakraborty, S. (2022). Probabilistic machine learning based predictive and interpretable digital twin for dynamical systems. arXiv preprint arXiv:2212.09240. [ArXiv](https://arxiv.org/abs/2212.09240)

# Schematic architecture of the proposed predictive digital twin framework for model updating of dynamical systems
![Architecture](Predictive_DT.png)

# Files
The main codes, described below, are standalone codes. They can be directly run. However, for the figures in the published article, one needs to run the files in the folder `Paper_figures`. A short despcription on the files are provided below for ease of readers.
  + `Example_1_BScholes.py` is the code to discover physics for the Example 1: Black-Scholes equation [article](https://arxiv.org/pdf/2208.05609.pdf).
  + `Example_2_Duffing.py` is the code to discover physics for the Example 2: Parametrically excited Duffing oscillator [article](https://arxiv.org/pdf/2208.05609.pdf).
  + `Example_3_2DOF.py` is the code to discover physics for the Example 3: 2DOF system [article](https://arxiv.org/pdf/2208.05609.pdf).
  + `Example_4_boucwen.py` is the code to discover physics for the Example 4: Bouc-Wen, with partially observed state variables [article](https://arxiv.org/pdf/2208.05609.pdf).
  - `utils_gibbs.py` is a part of gibbs sampling in section 2.2 [article](https://arxiv.org/pdf/2208.05609.pdf).
  * `utils_library.py` contains useful functions, like, library construction, data-normalization.
  + `utils_response.py` is the code to generate data using stochastic calculus.
The codes for the *Stochastic SINDy* are provided in the folder *Stochastic_Sindy*.

# BibTex
If you use any part our codes, please cite us at,
```
@article{tripura2022probabilistic,
  title={Probabilistic machine learning based predictive and interpretable digital twin for dynamical systems},
  author={Tripura, Tapas and Desai, Aarya Sheetal and Adhikari, Sondipon and Chakraborty, Souvik},
  journal={arXiv preprint arXiv:2212.09240},
  year={2022}
}
```
