#!/bin/bash
set -euo pipefail


pip install -U pip
pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
# PyTorch & Deepspeed
pip install torch==2.8.0 torchaudio torchvision "deepspeed>=0.17.2" pynvml
# FlashAttention / FlashInfer
pip install "flash-attn==2.8.3" --no-build-isolation
pip install flashinfer-python==0.3.1 --no-build-isolation

#pip install vllm==0.10.1
pip install "sglang[all]>=0.5.2" 
# Megatron & other GPU tools
pip install megatron-core==0.13.1 nvidia-ml-py



# pip install -e .[dev]
# Transformers & OpenAI SDK
# pip install transformers==4.56.0
# pip install openai==1.99.6

# cd
