#!/usr/bin/env sh
set -eu

# This helper installs code only. Dataset acquisition is deliberately separate:
# run the official alfworld-download CLI with an explicit --data-dir, then set
# ALFWORLD_DATA and ALFWORLD_TASK_MANIFEST before starting the service.
python -m pip install .
