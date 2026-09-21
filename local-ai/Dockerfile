# The local inference platform.
#
# Built on plain python-slim rather than an nvidia/cuda base image: the CUDA
# runtime that PyTorch needs ships inside its own wheels, so the only thing
# required from outside is the host driver, which the NVIDIA Container Toolkit
# mounts in. That keeps the image free of a second, conflicting CUDA install.
#
# No model weights are baked in. They are downloaded at runtime into a volume —
# a 34 GB layer would make this image unpushable and unversionable.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Blackwell (RTX 5090, sm_120) needs a CUDA 12.8 build. Installing torch first,
# and from that index, means the `.[gpu]` extra below finds its torch>=2.7
# requirement already satisfied and does not pull a default-index build over it.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=torch
# Its own layer, and by far the largest: ~3 GB of wheels that must not be
# re-fetched every time a line of Python changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX_URL} ${TORCH_VERSION}

# The whole tree, then the editable install. The package list in pyproject.toml
# is explicit, so setuptools rejects a metadata-only layer whose subpackage
# directories do not exist yet — the sources have to be here first. The pip
# cache mount keeps rebuilds cheap without baking the cache into the image.
COPY . .
# The dev extra is a few hundred KB and makes the image self-verifying:
#   docker compose run --rm local-ai pytest
RUN --mount=type=cache,target=/root/.cache/pip pip install -e '.[gpu,dev]'

# Defaults for the container. AI_ROOT is a volume; binding to 0.0.0.0 is safe
# here because only the published port is reachable from outside.
ENV AI_ROOT=/data \
    AI_HOST=0.0.0.0 \
    AI_PORT=8000 \
    AI_DEVICE=auto \
    AI_DTYPE=bfloat16 \
    HF_HOME=/data/cache

# The toolkit reads these to decide what to mount from the host driver.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/v1/system/health', timeout=4)"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
