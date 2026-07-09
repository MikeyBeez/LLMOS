#!/bin/bash
# Build llama-server with CUDA support for the RTX 5070 Ti (sm_120 / Blackwell).
# Single-arch build so it doesn't waste 10x the time on archs we don't have.
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
cd ~/llama.cpp
rm -rf build
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DBUILD_SHARED_LIBS=OFF \
    > /tmp/llamacpp_build.log 2>&1
cmake --build build --target llama-server -j 8 \
    >> /tmp/llamacpp_build.log 2>&1
echo DONE >> /tmp/llamacpp_build.log
