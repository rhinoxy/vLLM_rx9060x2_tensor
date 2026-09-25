#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/k-takeda/gAI-LLM/bin:$PATH"

cd "$DIR" && docker-compose logs -f "$@"
