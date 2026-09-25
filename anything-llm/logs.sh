#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/your_name/gAI-LLM/bin:$PATH"

cd "$DIR" && docker-compose logs -f "$@"
