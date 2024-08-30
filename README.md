<div align="center">

# Generative Recommenders

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)

Repicate the Generative Recommenders with Lightning and Hydra.

_Suggestions are always welcome!_

</div>

<br>

## Description

This repository aims to replicate the [Generative Recommenders](https://github.com/facebookresearch/generative-recommenders) using Lightning and Hydra. It hosts the code for the paper ["Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"](https://arxiv.org/abs/2402.17152). While primarily intended for personal learning, this repository offers several key features:

- **Efficient Training & Inference**: enhances training and inference speed by optimizing GPU utilization. As a result, the training of the MovieLens-1M dataset over 100 epochs can now be completed in under 10 minutes on a single 4090 or L4 machine.
- **Experimentation Made Easy**: Easily manage and create hierarchical configurations with overrides via config files and command-line options to support various experiments.
- **Modular Configuration**: Dynamically instantiate objects through configuration files, allowing seamless switching between different datasets or modules without extensive rewriting.
- **Hardware Agnostic**: The dependency on NVIDIA GPUs has been removed, enabling you to run the scripts on any device, including local machines for training, evaluation, and debugging.
- **Improved Readability**: The code has been significantly refactored for clarity. The Generative Recommenders module is now divided into four major components: embeddings, preprocessor, sequence encoder, and postprocessor, making the training and evaluation processes more transparent.

## Installation

It is recommended to use `uv` to install the library:

```bash
uv venv -p 3.10 && source .venv/bin/activate
uv pip install --extra dev --extra test -r pyproject.toml
uv pip install -e . --no-deps
```

For Linux systems with GPU support, you can also install `fbgemm-gpu` to enhance performance:

```bash
uv pip install fbgemm-gpu==0.7.0
```

## How to Run

Prepare dataset based on config.

```bash
make prepare_data data=ml-1m
```

Train the Model with Default Configuration

```bash
# Train on CPU
make train trainer=cpu

# Train on GPU
make train trainer=gpu
```

Train the Model with a Specific Experiment Configuration. Choose an experiment configuration from [configs/experiment/](configs/experiment/):

```bash
make train experiment=ml-1m-hstu
```

Evaluate the Model with a Given Checkpoint

```bash
make eval ckpt_path=example_checkpoint.pt
```

Override Parameters from the Command Line

```bash
make train trainer.max_epochs=20 data.batch_size=64
```

## Experiment Result:

### MovieLens-1M (ML-1M)

To ensure reproducibility and eliminate randomization, the sample in the dataset generation was removed, and the seed in training was set to 42.

| Method | HR@10  | NDCG@10 | HR@50  | NDCG@50 | HR@100 | NDCG@100 | HR@200 | NDCG@200 |
| ------ | ------ | ------- | ------ | ------- | ------ | -------- | ------ | -------- |
| HSTU   | 0.2975 | 0.1680  | 0.5815 | 0.2308  | 0.6887 | 0.2483   | 0.7735 | 0.2602   |

Feel free to explore and modify the configurations to suit your needs. Your contributions and suggestions are always welcome!
