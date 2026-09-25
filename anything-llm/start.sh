#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/k-takeda/gAI-LLM/bin:$PATH"

echo "Starting AnythingLLM..."
cd "$DIR" && docker-compose up -d

echo ""
echo "AnythingLLM is starting up!"
echo "Web UI: http://localhost:3001"
