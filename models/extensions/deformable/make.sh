#!/bin/bash
CXX=g++ TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python -m pip install .
rm -rf build Deformable.egg-info dist
