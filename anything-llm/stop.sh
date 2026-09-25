#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/k-takeda/gAI-LLM/bin:$PATH"

echo "Stopping AnythingLLM..."
cd "$DIR" && docker-compose down
