#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/k2-horizon-exl3}"
PY="$WORK_DIR/.venv/bin/python"
MODEL_DIR="$WORK_DIR/models/6.50bpw"
OVERLAY="$WORK_DIR/biases/k2-routing-bias-overlay.safetensors"
"$PY" "$RECIPE_DIR/scripts/verify_assets.py" --model-dir "$MODEL_DIR" --overlay "$OVERLAY"
"$PY" "$RECIPE_DIR/scripts/patch_tabby.py" "$WORK_DIR/tabbyAPI"
# Local-only benchmark endpoint, never reverse-proxy as-is.
"$PY" - "$WORK_DIR/config.yml" "$WORK_DIR/models" <<'PY'
from pathlib import Path
import sys
path, models = map(Path, sys.argv[1:])
path.write_text('network:\n  host: 127.0.0.1\n  port: 5001\n  disable_auth: true\n  allowed_origins: []\n  disable_fetch_requests: true\nmodel:\n  model_dir: ' + str(models) + '\n  model_name: 6.50bpw\n  backend: exllamav3\n  max_seq_len: 4096\n  cache_size: 4096\n  max_batch_size: 1\n  tensor_parallel: false\n  cache_mode: FP16\ndraft_model:\n  draft_mode: disabled\n', encoding='utf-8')
PY
export K2_ROUTING_BIAS_OVERLAY="$OVERLAY"
cd "$WORK_DIR/tabbyAPI"
exec "$PY" main.py --config "$WORK_DIR/config.yml"
