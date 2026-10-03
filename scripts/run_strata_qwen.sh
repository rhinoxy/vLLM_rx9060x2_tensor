#!/bin/bash
# Strata Qwen 3.8 Flash Next (IQ3_S) Launch Script with NUMA Interleaving
# Optimized for Dual RX 9060 XT (gfx1200) across NUMA nodes

STRATA_DIR="/home/your_name/Strata"
CONFIG_FILE="${STRATA_DIR}/strata-iq3_s.json"
PORT="8081"

if [ ! -d "$STRATA_DIR" ]; then
    echo "Error: Strata directory not found at $STRATA_DIR"
    exit 1
fi

cd "$STRATA_DIR" || exit 1

echo "=========================================================="
echo "Starting Strata: Qwen 3.8 Flash Next (IQ3_S)"
echo "NUMA policy: --interleave=all (Multi-socket memory striping)"
echo "Port: $PORT"
echo "=========================================================="

exec numactl --interleave=all \
  "${STRATA_DIR}/.venv/bin/python" \
  "${STRATA_DIR}/serve/server.py" \
  --engine strata \
  --config "$CONFIG_FILE" \
  --port "$PORT" \
  --open
