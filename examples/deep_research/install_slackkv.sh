#!/usr/bin/env bash
# Recreate the research client from the checked-in lock, with the local SDK.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${UV_CACHE_DIR:?Set UV_CACHE_DIR to a data disk before installation}"
uv sync --frozen --python 3.12
.venv/bin/python -c 'import deepagents, langchain_openai; print("CLIENT_IMPORT_OK", deepagents.__file__)'
