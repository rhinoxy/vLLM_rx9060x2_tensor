#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-status}"

case "$ACTION" in
  qwen|qwen3.8)
    echo "Switching to Qwen 3.8 27B..."
    bash "$SCRIPT_DIR/run_vllm_qwen.sh"
    ;;
  gemma|gemma4)
    echo "Switching to Gemma 4 26B..."
    bash "$SCRIPT_DIR/run_vllm_gemma.sh"
    ;;
  stop)
    bash "$SCRIPT_DIR/stop_vllm.sh"
    ;;
  status)
    echo "=== vLLM Container Status ==="
    docker ps --filter "name=vllm-server" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== API Health Check ==="
    curl -s --connect-timeout 2 http://localhost:8001/health 2>&1 || echo "API server not reachable on port 8001"
    echo ""
    ;;
  *)
    echo "Usage: $0 {qwen|gemma|stop|status}"
    exit 1
    ;;
esac
