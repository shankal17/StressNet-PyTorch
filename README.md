# Stress Field Prediction in Cantilevered Structures Using Convolutional Neural Networks

<p align="left"> 
    <a href="https://www.python.org/">
        <img  src="https://img.shields.io/badge/python-3.8-blue"/></a> 
    <a href="https://pytorch.org/">
        <img  src="https://img.shields.io/badge/pytorch-1.7.1-blue"/></a> 
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green"></a>
    <a href="https://doi.org/10.1115/1.4044097">
        <img src="https://img.shields.io/badge/DOI-10.1115/1.4044097-orange"></a>
</p>

This repository contains a PyTorch implementation of the original StressNet paper (🔗 [Nie et al.](https://doi.org/10.1115/1.4044097)). 

## Table of Contents

- [Performance](#Performance)
- [Citation](#Citation)

## Performance

This implementation has been tested on Windows 10 with a Nvidia RTX 2080 GPU.

### Error Metrics

#### For Training

| Model                | MSE   | MAE  | MRE (%) |
| -------------------- | ----- | ---- | ------- |
| SCSNet (Original)    | 83.63 | 4.28 | 10.40   |
| SCSNet (PyTorch)     |       |      |         |
| StressNet (Original) | 0.14  | 0.22 | 1.99    |
| StressNet (PyTorch)  |       |      |         |

#### For Testing

| Model                | MSE   | MAE  | MRE (%) |
| -------------------- | ----- | ---- | ------- |
| SCSNet (Original)    | 84.07 | 4.30 | 10.43   |
| SCSNet (PyTorch)     |       |      |         |
| StressNet (Original) | 0.15  | 0.23 | 2.04    |
| StressNet (PyTorch)  |       |      |         |

## Citation

The original implementation of StressNet is available in [Tensorflow 1.4](https://github.com/zhenguonie/stress_net).

To cite the original paper, please using the following BibTex entry.

```
@article{nie2020stress,
  title={Stress field prediction in cantilevered structures using convolutional neural networks},
  author={Nie, Zhenguo and Jiang, Haoliang and Kara, Levent Burak},
  journal={Journal of Computing and Information Science in Engineering},
  volume={20},
  number={1},
  year={2020},
  publisher={American Society of Mechanical Engineers Digital Collection}
}
```