#!/usr/bin/env bash
# Run three NVIDIA Newton examples without opening a graphics window.
# Use this to prove that Newton and CUDA work before presenting GUI output.

set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$project_dir/.venv-newton/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "Newton is not installed in $project_dir/.venv-newton."
    exit 1
fi

for example in basic_pendulum basic_shapes basic_joints; do
    echo "Running: $example"
    "$python_bin" -m newton.examples "$example" --device cuda:0 --viewer null --num-frames 120
done

echo "All Newton examples completed successfully."
