#!/bin/bash
echo "Stopping vLLM server..."
docker stop vllm-server 2>/dev/null || true
docker rm -f vllm-server 2>/dev/null || true
echo "vLLM server stopped."
