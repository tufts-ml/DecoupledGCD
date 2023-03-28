# CCGaussian

[![CCGaussian](https://github.com/tufts-ai-robotics-group/CCGaussian/actions/workflows/main.yml/badge.svg)](https://github.com/tufts-ai-robotics-group/CCGaussian/actions/workflows/main.yml)

Experimental class-conditional Gaussian models.

The dependencies can be installed within a Pipenv with the following commands:
```
pipenv install --categories "packages torch_cpu"
```
PyTorch may require different versions depending on the machine it is running on. The default command is for non-CUDA machines while swapping `torch_cpu` for `torch_cu117` installs PyTorch for CUDA 11.7. If a non-default version of PyTorch is required then generate the appropriate Pip command on the [PyTorch website](https://pytorch.org/get-started/locally/) then run it within the Pipenv by prepending ```pipenv run``` to it.
