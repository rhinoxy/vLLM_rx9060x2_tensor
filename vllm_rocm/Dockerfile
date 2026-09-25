# Base image: ROCm vLLM
FROM rocm/vllm:latest

LABEL maintainer="rhinoxy"
LABEL description="ROCm vLLM custom image for AMD Radeon RX 9060 XT (gfx1200 / Navi)"

# Install GGUF runtime plugins with --no-deps to prevent overwriting container-optimized libraries (e.g., tensorizer 2.12.1)
RUN pip install --no-cache-dir --no-deps gguf vllm-gguf-plugin

# Apply all ROCm / Navi compatibility patches during build
COPY patch_transformers.py /tmp/patch_transformers.py
RUN python3 /tmp/patch_transformers.py && rm /tmp/patch_transformers.py

# Healthcheck configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1
