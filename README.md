# Bregman Learning Framework for Sparse Neural Networks

This repository extends the [BregmanLearning repository](https://github.com/TimRoith/BregmanLearning) to explore structured sparsity in neural networks using optimisation-based methods. The key techniques include Linearised Bregman Iteration (LinBreg), nuclear norm regularisation, momentum-based variants, and dynamic scheduling strategies. These methods are evaluated on standard vision datasets such as MNIST and Fashion-MNIST.

---

##  Overview

The project implements a Bregman learning framework to train sparse neural networks through inverse scale space methods. Unlike magnitude pruning, the approach adds important weights gradually via a proximal mapping. Features include:

- LinBreg, AdaBreg, and PGD variants
- ℓ₁, group, and nuclear norm regularization
- Dynamic rank scheduling for nuclear norm
- Comparison between full vs truncated SVD
- Baseline and momentum-accelerated versions

---

## Repository Structure

```
models/
├── mnist_conv.py            # CNN for MNIST and Fashion-MNIST
├── fully_connected.py       # MLP with configurable layers
└── aux_funs.py              # Sparsity-related utilities

notebooks/
├── MLP-Classification.ipynb
└── ConvNet-Classification.ipynb

utils/
├── configuration.py         # Experiment configuration and runner
└── datasets.py              # Dataset loading and preprocessing

custom_regularizers.py       # Nuclear norm with dynamic rank handling
regularizers.py              # L1, group, and other convex regularizers
optimizers.py                # LinBreg, AdaBreg, ProxSGD, PGD

train.py                     # Training and validation routines
requirements.txt             # Python dependencies
```

---

## Installation

Install with pip:

```bash
pip install -r requirements.txt
```

Ensure Python 3.8+ and PyTorch are properly set up.

---



## Reproducing Dissertation Results

| Section | Experiment Description                          | Relevant Components                     |
|---------|--------------------------------------------------|------------------------------------------|
| 5.1     | LinBreg + Nesterov Momentum                     | `optimizers.py`, `train.py`              |
| 5.2     | Nuclear vs ℓ₁ vs Group sparsity                 | `regularizers.py`, `custom_regularizers.py` |
| 5.3     | Dynamic rank scheduling                         | `DynamicRankNuclearRegularizer` class    |
| 5.4     | Truncated vs Full SVD comparison                | Uses `torch.svd_lowrank`, `torch.svd`    |

Each experiment can be configured using the `Conf` class. Logs are saved based on `conf.name`.

---

## Reference

If you use this code or build upon it, please cite:

```bibtex
@article{bungert2022bregman,
  title={A Bregman Learning Framework for Sparse Neural Networks},
  author={Bungert, Leon and Roith, Tim and Tenbrinck, Daniel and Burger, Martin},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={1},
  pages={1--43},
  year={2022}
}
```

---
