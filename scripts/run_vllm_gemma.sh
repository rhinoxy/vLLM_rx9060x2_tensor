#!/bin/bash
docker stop vllm-server 2>/dev/null || true
docker rm -f vllm-server 2>/dev/null || true

echo "=== Starting vLLM: Gemma 4 26B on Port 8001 (TP=2) ==="

docker run -d \
  --name vllm-server \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --network=host \
  --ipc=host \
  --group-add sudo \
  -v /home/k-takeda/gAI-LLM/models:/models:ro \
  -v /usr/share/ollama:/usr/share/ollama:ro \
  -v /home/k-takeda/gAI-LLM/vllm_rocm/patch_transformers.py:/tmp/patch_transformers.py:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/tvm-ffi:/root/.cache/tvm-ffi \
  -e GLOO_SOCKET_IFNAME=lo \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --restart unless-stopped \
  rocm-vllm:custom \
  bash -c "python3 /tmp/patch_transformers.py && exec vllm serve /models/gemma4-26b/model.gguf \
    --tokenizer google/gemma-4-26B-A4B-it \
    --hf-config-path /models/gemma4-26b \
    --served-model-name gemma4:26b \
    --tensor-parallel-size 2 \
    --port 8001 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --enforce-eager"

echo "vLLM server started in background as 'vllm-server'."
echo "To view logs: docker logs -f vllm-server"
echo "To stop: /home/k-takeda/OpenClaw/stop_vllm.sh"
