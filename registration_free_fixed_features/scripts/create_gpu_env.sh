#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
environment=${project_root}/registration_free_fixed_features/.venv

uv venv "${environment}" --python 3.11
uv pip install \
    --python "${environment}/bin/python" \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install \
    --python "${environment}/bin/python" \
    --requirement "${project_root}/registration_free_fixed_features/requirements.txt"

"${environment}/bin/python" -c \
    'import torch; print({"torch": torch.__version__, "cuda_build": torch.version.cuda})'
