# Copyright (c) 2026 oLV - Olivier LAVAUD
# SPDX-License-Identifier: MIT
#
# Gradio playground + JSON API for Pocket TTS (pocket_tts.py).
#
#   docker build -f Dockerfile.app -t pocket-tts-app .
#   docker run --rm -p 7860:7860 pocket-tts-app --language french
#
# The model weights are downloaded from HuggingFace on startup, so mount the
# caches (see docker-compose.yaml) or accept the download at every start.
# Pass `-e HF_TOKEN=...` to use the voice-cloning weights of the gated repo.
FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    HF_HOME=/root/.cache/huggingface \
    POCKET_TTS_HOST=0.0.0.0 \
    POCKET_TTS_PORT=7860

# Dependencies first: this layer is only invalidated when the lockfile changes.
# --no-dev skips the training and evaluation tools, and --frozen keeps the
# committed lock as-is. Gradio is not a declared dependency, so it is installed
# on top of the synced environment.
COPY ./pyproject.toml ./uv.lock ./README.md ./.python-version ./
RUN uv sync --no-dev --frozen --no-install-project

COPY ./pocket_tts ./pocket_tts
COPY ./pocket_tts.py ./pocket_tts.py
# Gradio is installed after the sync: `uv sync` would remove anything that is not
# in the lockfile, and gradio is deliberately not a declared dependency.
RUN uv sync --no-dev --frozen \
    && uv pip install --python /app/.venv/bin/python gradio soundfile

EXPOSE 7860
# Called directly rather than through `uv run`, which would re-sync and drop the
# gradio installed above.
ENTRYPOINT ["/app/.venv/bin/python", "pocket_tts.py"]