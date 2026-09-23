#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-$HOME/k2-horizon-exl3}"
HF="$WORK_DIR/.venv/bin/hf"
[[ -x "$HF" ]] || { echo 'Run scripts/install.sh first (hf CLI missing).' >&2; exit 1; }
mkdir -p "$WORK_DIR/models"
"$HF" download vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3 \
  --revision c88277ce7f6b90b723f79b5188c0f1b951732099 \
  --include '6.50bpw/*' --local-dir "$WORK_DIR/models"
echo "Downloaded $WORK_DIR/models/6.50bpw"
