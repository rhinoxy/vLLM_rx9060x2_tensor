FROM rocm/vllm:latest

RUN pip install --no-cache-dir --no-deps gguf vllm-gguf-plugin

COPY patch_transformers.py /tmp/patch_transformers.py
RUN python3 /tmp/patch_transformers.py && rm /tmp/patch_transformers.py
