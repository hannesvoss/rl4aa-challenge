<!-- [![DOI](https://zenodo.org/badge/700362904.svg)](https://zenodo.org/doi/10.5281/zenodo.10886639) -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Reinforcement Learning Challenge at the RL4AA'25 Workshop

This repository contains the material for challenge of the [RL4AA'25](https://indico.scc.kit.edu/event/4216/overview) workshop.

Homepage for RL4AA Collaboration: [https://rl4aa.github.io/](https://rl4aa.github.io/)

## Disclaimer &#x2757;

This repository contains advanced Python tutorials developed with care and dedication to foster learning and collaboration. The code and materials provided here are the result of significant effort, including state-of-the-art research and unpublished or pre-peer-reviewed work.

We share these resources in good faith, aiming to contribute to the community and advance knowledge in our field. If you use or build upon any part of this tutorial—whether in research, software, or educational materials—proper citation is required. Please cite the tutorial as indicated in the repository or its associated Zenodo entry.

While we encourage reuse and adaptation of our work, uncredited use or plagiarism is unacceptable. We actively monitor citations and expect users to engage in responsible scholarly practice. Failure to properly attribute this work may lead to formal actions.

By using this repository, you acknowledge and respect the effort behind it. We appreciate your support in maintaining academic integrity and fostering an open, collaborative environment.

Happy coding, and thank you for citing responsibly! 😊

## Citing the materials

<!--
This tutorial is uploaded to [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10886639).
Please use the following DOI when citing this code:

```bibtex
@software{hirlaender_2024_10887397,
    title        = {{Tutorial on Meta-Reinforcement Learning and GP-MPC at the RL4AA'24 Workshop}},
    author       = {Hirlaender, Simon and Kaiser, Jan and Xu, Chenran and Santamaria Garcia, Andrea},
    year         = 2024,
    month        = mar,
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.10887397},
    url          = {https://doi.org/10.5281/zenodo.10887397},
    version      = {v1.0.2}
} -->

## Getting started

- You will require about **1 GB of free disk space** &#x2757;
- Make sure you have Git installed in your terminal &#x2757;

Start by cloning locally the repository of the challenge by running this command in your terminal:

```bash
git clone https://github.com/RL4AA/rl4aa25-challenge.git
```

### Installing virtual environment

#### Using conda-forge

- If you don't have conda installed already, you can install the `miniforge` as described in the [GitHub repository](https://github.com/conda-forge/miniforge) or download from the [conda-forge site](https://conda-forge.org/download/). Once `miniforge` is installed, you can use the `conda` commands as usual.
- We recommend installing `miniforge` the day beforehand to avoid network overload during the challenge &#x2757; &#x2757;

**Note**: If you already have anaconda or miniconda installed, please remove the anaconda channels from the channel list and use `conda-forge` (community version), as the package installation from commercial channels is blocked insde of the DESY network.

Once `miniforge` is installed run this command in your terminal:

```bash
conda env create -f environment.yaml
```

This should create a virtual environment named `rl4aa25-challenge` and install the necessary packages inside.

Afterwards, activate the environment using

```bash
conda activate rl4aa25-challenge
```

### Getting to know the ARES EA Environment

Next, open the Jupyter notebook `challenge_introduction.ipynb` either in your editor or in terminal via

```bash
jupyter notebook
```

## Folder Structure

- `src` Contains the source code for the RL environment and the GP-MPC controller
  - `src/environments/ea` contains the gymnasium environment for the ARES-EA transverse tuning task
  - `src/wrappers` contains custom wrappers for the EA environment
  - `src/train` contains scripts to train a PPO agent to solve the task
- `data` contains the data from evaluating your agents
  - `data/csvs` contains the CSV files generated by the evaluation scripts for challenge submission

## Further Resources

For more examples and details on the ARES RL environment, c.f.

- [Reinforcement learning-trained optimisers and Bayesian optimisation for online particle accelerator tuning](https://www.nature.com/articles/s41598-024-66263-y)
  - Code repository: <https://github.com/desy-ml/rl-vs-bo>
- [Learning-based Optimisation of Particle Accelerators Under Partial Observability Without Real-World Training](https://proceedings.mlr.press/v162/kaiser22a.html)
