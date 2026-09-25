#!/bin/bash
TARGET=${1:-vllm-server}

echo "=== Stopping vLLM server ($TARGET) ==="
if docker ps -q --filter "name=$TARGET" | grep -q .; then
  docker stop "$TARGET"
  docker rm -f "$TARGET" 2>/dev/null || true
  echo "✅ $TARGET stopped and removed successfully."
else
  echo "ℹ️  No running container named $TARGET."
fi
