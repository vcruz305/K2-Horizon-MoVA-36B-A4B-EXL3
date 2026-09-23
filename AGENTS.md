# AGENTS.md — K2-Horizon-MoVA-36B-A4B-EXL3

## Purpose
This is a GPU-agnostic, Linux/NVIDIA TabbyAPI + ExLlamaV3 recipe for Victor Cruz's K2-Horizon-MoVA-36B-A4B 6.50bpw EXL3 pack. It is not a weight mirror or a DGX Spark-only recipe. The only measured host to date is one GB10; never extrapolate its speed to other GPUs.

## Source-of-truth pins
- Quant: `vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3`, revision `c88277ce7f6b90b723f79b5188c0f1b951732099`, folder `6.50bpw/` only. Four shards total `31,288,045,811` bytes.
- K2-capable runtime fork: `vcruz305/exllamav3` commit `be01a273244ae4238e2919aecdc86bc08d7fb154` (merged PR #10). Do not substitute stock upstream or an unmerged optimization branch.
- TabbyAPI: `theroyallab/tabbyAPI` commit `f07131cd8fe34e449fe87cdd3a066b52b96d3cac`.
- Bias source: public `IFM/K2-Horizon-MoVA-36B-A4B` revision `7730b92d1b574e04663b04023d5d6fa83475432f`. The fork extractor produces a separate 90-tensor BF16 routing-bias overlay with SHA-256 `8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c`.
Never zero-fill missing biases, modify the quant shards, substitute another overlay, or silently float a pin. Model artifacts, overlays, virtualenvs, secrets, and benchmark raw logs stay outside this Git repository.

## Files and workflow
- `README.md`: user-facing requirements, setup, smoke, measured baseline, limits, and troubleshooting.
- `scripts/install.sh`: exact source pins, Python 3.12, compatible CUDA-enabled Torch and toolkit, and `uvloop` (required by the pinned TabbyAPI on the tested Linux host). No hard-coded GPU architecture; other GPUs need a matching driver/toolkit/wheel and sufficient memory.
- `scripts/download.sh`: only the pinned `6.50bpw/*` files into the external `WORK_DIR`.
- `scripts/extract_overlay.sh`: audited pinned fork extractor; validate the overlay digest before serving.
- `scripts/verify_assets.py`: fail-closed architecture, index, shard-size, 90-tensor shape/name, and overlay digest checks. A size check is not a shard SHA-256 check.
- `scripts/patch_tabby.py`: narrowly patches exactly one callsite in the pinned TabbyAPI checkout to pass the overlay. Reject unknown revisions or an unexpected callsite.
- `scripts/serve.sh`: validates assets, writes a local config outside Git, and binds unauthenticated benchmarking to `127.0.0.1:5001` only. Never expose or reverse-proxy it without separate authentication and security review.
- `scripts/test_recipe.py`: offline regression tests, no GPU or model download required.
- `LICENSE` and `THIRD_PARTY_NOTICES.md`: separate this recipe's license from the model and runtime licenses.

From the recipe root on a supported Linux NVIDIA host with Python 3.12, `nvcc`, a matching CUDA driver and enough disk/VRAM or unified memory:

```bash
bash scripts/install.sh
bash scripts/download.sh
bash scripts/extract_overlay.sh
"${WORK_DIR:-$HOME/k2-horizon-exl3}/.venv/bin/python" scripts/verify_assets.py \
  --model-dir "${WORK_DIR:-$HOME/k2-horizon-exl3}/models/6.50bpw" \
  --overlay "${WORK_DIR:-$HOME/k2-horizon-exl3}/biases/k2-routing-bias-overlay.safetensors"
bash scripts/serve.sh
```

In another terminal, check `/v1/models` and a bounded `/v1/chat/completions` call per README. Fail closed on an unavailable wheel, invalid overlay, wrong model ID, unknown source callsite, or OOM rather than silently changing the model or opening the listener to the network.

## Change and publication gates
1. Run `python scripts/test_recipe.py`, `python -m compileall -q scripts`, `bash -n scripts/*.sh`, and `git diff --check`. These are static/CPU checks, not proof of a GPU load.
2. Runtime/pin changes require a real bounded model completion and unload on a permitted GPU, with exact source/extension identities and overlay digest. No other person's GPU process may be stopped.
3. Benchmark claims must name GPU, workload, sampling, concurrency, warmup and sample count, and per-stream versus aggregate metrics. The existing Sixcat v0.7.0 GB10 C=1 decode p50 of 19.28 tok/s is not universal. Do not claim the unmerged mixed-K branch or this new install script produced that prior result.
4. Do not commit weights, credentials, tokens, generated overlays or host-specific private paths. Do not fabricate data. Preserve model/TabbyAPI/ExLlamaV3 attribution.
5. Publish only to the authorized public GitHub repo `vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3`. Verify the default-branch commit and exact file inventory from GitHub before calling it shipped. Never create a private repo or a DGX-Spark-suffixed alternate.
