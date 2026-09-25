#!/bin/bash
set -e

# Stop any currently running vLLM server (mutually exclusive due to 32GB VRAM limits)
echo "=== Stopping previous vLLM server instances ==="
docker stop vllm-server 2>/dev/null || true
docker rm -f vllm-server 2>/dev/null || true

echo "=== Starting vLLM: Qwen 3.8 27B on Port 8001 (TP=2) ==="

RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3 || echo 992)

docker run -d \
  --name vllm-server \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add "$RENDER_GID" \
  --network=host \
  --ipc=host \
  -v /home/k-takeda/gAI-LLM/models:/models:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/tvm-ffi:/root/.cache/tvm-ffi \
  -e GLOO_SOCKET_IFNAME=lo \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_PROTO=Simple \
  -e VLLM_ROCM_USE_AITER=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --restart unless-stopped \
  rocm-vllm:custom \
  vllm serve /models/qwen3.8-27b/model.gguf \
    --tokenizer Qwen/Qwen3.8-27B \
    --chat-template /models/template_qwen.jinja \
    --hf-config-path /models/qwen3.8-27b \
    --served-model-name qwen3.8:27b \
    --tensor-parallel-size 2 \
    --port 8001 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --enforce-eager

echo "vLLM server started in background as 'vllm-server'."
echo "To monitor startup: docker logs -f vllm-server"
echo "To check health:   curl -s http://localhost:8001/health"
echo "To stop:           bash scripts/stop_vllm.sh"
