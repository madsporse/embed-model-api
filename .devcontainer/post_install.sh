#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y build-essential git curl
mkdir -p /workspaces/.cache/huggingface /workspaces/.cache/sentence-transformers

# Poetry config and install
poetry --version
poetry config virtualenvs.in-project true
if [ -f "pyproject.toml" ]; then
  poetry install --no-interaction --no-ansi
fi

echo "Post-install complete"
