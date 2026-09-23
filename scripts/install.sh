#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-$HOME/k2-horizon-exl3}"
EXL_SHA=be01a273244ae4238e2919aecdc86bc08d7fb154
TABBY_SHA=f07131cd8fe34e449fe87cdd3a066b52b96d3cac
TORCH_SPEC="${TORCH_SPEC:-torch==2.13.0+cu130}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
[[ "$(uname -s)" == Linux ]] || { echo 'Requires Linux with a supported NVIDIA GPU.' >&2; exit 1; }
command -v nvcc >/dev/null || { echo 'A compatible CUDA nvcc is required.' >&2; exit 1; }
command -v python3.12 >/dev/null || { echo 'python3.12 required.' >&2; exit 1; }
command -v git >/dev/null || { echo 'git required.' >&2; exit 1; }
mkdir -p "$WORK_DIR"
clone_pin() {
    local url="$1" dir="$2" pin="$3"
    if [[ ! -e "$dir" ]]; then git clone "$url" "$dir"; fi
    [[ -d "$dir/.git" ]] || { echo "Not a Git checkout: $dir" >&2; exit 1; }
    [[ "$(git -C "$dir" remote get-url origin)" == "$url" ]] || { echo "Wrong remote: $dir" >&2; exit 1; }
    if [[ "$(git -C "$dir" rev-parse HEAD)" != "$pin" ]]; then
        [[ -z "$(git -C "$dir" status --porcelain)" ]] || { echo "Dirty checkout: $dir" >&2; exit 1; }
        git -C "$dir" fetch origin "$pin"
        git -C "$dir" checkout --detach "$pin"
    fi
    [[ "$(git -C "$dir" rev-parse HEAD)" == "$pin" ]] || exit 1
}
clone_pin https://github.com/vcruz305/exllamav3.git "$WORK_DIR/exllamav3" "$EXL_SHA"
clone_pin https://github.com/theroyallab/tabbyAPI.git "$WORK_DIR/tabbyAPI" "$TABBY_SHA"
if [[ ! -e "$WORK_DIR/.venv/bin/python" ]]; then python3.12 -m venv "$WORK_DIR/.venv"; fi
PY="$WORK_DIR/.venv/bin/python"
"$PY" -m pip install -U pip setuptools wheel ninja
"$PY" -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
# Compile the fork against the installed CUDA-enabled Torch; no fixed GPU arch.
"$PY" -m pip install --no-build-isolation "$WORK_DIR/exllamav3"
# Base TabbyAPI install deliberately excludes its stock-wheel CUDA extras.
"$PY" -m pip install "$WORK_DIR/tabbyAPI" 'huggingface_hub[cli]' requests uvloop
"$PY" -c 'import torch, exllamav3, uvloop; assert torch.version.cuda and torch.cuda.is_available(); print("CUDA, fork and uvloop imports OK:", torch.__version__)'
echo "Installed exact runtime source pins under $WORK_DIR"
