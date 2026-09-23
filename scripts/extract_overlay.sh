#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-$HOME/k2-horizon-exl3}"
EXL_SHA=be01a273244ae4238e2919aecdc86bc08d7fb154
[[ "$(git -C "$WORK_DIR/exllamav3" rev-parse HEAD)" == "$EXL_SHA" ]] || { echo 'Wrong ExLlamaV3 commit.' >&2; exit 1; }
PY="$WORK_DIR/.venv/bin/python"
"$PY" "$WORK_DIR/exllamav3/tools/extract_k2_routing_bias_overlay.py" \
  --output-dir "$WORK_DIR/biases"
"$PY" "$(dirname "$0")/verify_assets.py" --overlay "$WORK_DIR/biases/k2-routing-bias-overlay.safetensors"
