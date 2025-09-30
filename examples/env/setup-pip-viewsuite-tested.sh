#!/bin/bash
set -euo pipefail


pip install -U pip
pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
# PyTorch & Deepspeed
pip install deepspeed==0.17.6 pynvml==13.0.1
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 "deepspeed>=0.17.2" pynvml
# FlashAttention / FlashInfer
pip install "flash-attn==2.8.3" --no-build-isolation
pip install flashinfer-python==0.2.11.post3 --no-build-isolation

#pip install vllm==0.10.1

# Megatron & other GPU tools
pip install megatron-core==0.13.1
pip install git+https://github.com/garrett4wade/cugae@f0c7198cb3e7265f43218a65ea3db4982520dd08 --no-build-isolation

# Transformers & OpenAI SDK
pip install transformers==4.56.0
pip install openai==1.99.6


pip install -e .[dev]
pip install "sglang[all]<=0.5.1"

pip install -e evaluation/latex2sympy
