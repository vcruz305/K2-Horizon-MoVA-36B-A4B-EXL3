#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-$HOME/k2-horizon-exl3}"
PACK="${PACK:?Set PACK to 8.00, 6.50, 5.00, 4.00, 2.50 or 2.00}"
case "$PACK" in
  8.00|6.50|5.00|4.00|2.50|2.00) ;;
  *) echo "Unsupported PACK: $PACK" >&2; exit 1 ;;
esac
HF="$WORK_DIR/.venv/bin/hf"
[[ -x "$HF" ]] || { echo 'Run scripts/install.sh first (hf CLI missing).' >&2; exit 1; }
mkdir -p "$WORK_DIR/models"
"$HF" download vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3 \
  --revision db645b888e6e05e89e7922230e0c1d526f87615c \
  --include "${PACK}bpw/*" --local-dir "$WORK_DIR/models"
echo "Downloaded $WORK_DIR/models/${PACK}bpw"
