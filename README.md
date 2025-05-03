# VisionTransformers (ViT)

ViTs with absolute positional encoding (APE) and relative positional encoding (RPE) are implemented in [models.py](vit_utils/models.py). ViTs with rotary positional encoding (RoPE) are implemented in [models_rope.py](vit_utils/models_rope.py).

Baseline models with APE, RPE and RoPE variants:
  - `vit_ape`
  - `vit_ape_reg_rpe`
  - `vit_ape_poly_rpe`
  - `vit_ape_axial_rope`
  - `vit_ape_mixed_rope`

## Installation
An environment with Python 3.8 or higher version is required, as well as `pip` package.

```
git clone https://github.com/xchen2763/VisionTransformers

cd VisionTransformers

pip install -r requirements.txt
```

Note that a `torch` and `torchvision` version different from `requirements.txt` might be required, which depends on the `CUDA Toolkit` version of your environment.

## Quickstart
Run [run_cifar10.sh](./run_cifar10.sh) for training and testing ViT models with different APE and RPE variants on CIFAR-10 dataset. Run [run_mnist.sh](./run_mnist.sh) for training and testing models on MNIST dataset. Run [plot.ipynb](./plot.ipynb) for visualization of training/testing losses and evaluation accuracy. All the results including logs, models and plots will be in `output` directory.

## Reference
This project uses code from [rope-vit](https://github.com/naver-ai/rope-vit), licensed under the Apache License 2.0.