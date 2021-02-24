# Scalable Computations of Wasserstein Barycenter viaInput Convex Neural Networks

Jiaojiao Fan, [Amirhossein Taghvaei](https://amirtag.github.io/), [Yongxin Chen](https://yongxin.ae.gatech.edu/)
[[arXiv](https://arxiv.org/abs/2007.04462)]

## Citation

```
@misc{jiao2021nwb,
title={Scalable Computations of Wasserstein Barycenter viaInput Convex Neural Networks},
author={Jiaojiao Fan},
year={2021}
}
```

## Dependencies and Installation

1. The following python packages are required:

- pytorch (>= 1.3.1)
- [GPUtil](https://github.com/anderskm/gputil)
- sklearn
- searborn
- POT
- matplotlib
- [jacinle](https://github.com/vacancy/Jacinle)
- [pytorch_fid](https://github.com/mseitzer/pytorch-fid)

2. [git large file storage](https://git-lfs.github.com/) has to be initialized to download the input data.

## Code structure

- The scripts in the root such as `G2G_sameW_3loop.py` are the core code for our NWB (Neural Wasserstein Barycenter) implementation. For example, if you are interested in Gaussian example, run in terminal `python G2G_sameW_3loop.py --parameter_1 param1_value --parameter_2 param2_value`.

- `generator_example/` contains the scripts to generate comparison results or visualization. For example, if you are interested in Gaussian example, run in terminal `python ./generator_example/NWB1_gmm.py --parameter_1 param1_value --parameter_2 param2_value`.

- `optimal_transport_modules/` contains the auxiliary utility modules.

## config file

The configuration of an experiment is entirely described by a `optimal_transport_modules/cfg.py` config file. If you want to change the parameter, please change them there.

## What if you find a bug?

This repository is still under construction. If you meet a bug when you run the code, please raise up an issue, thank you!
