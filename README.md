# K2-Horizon-MoVA-36B-A4B EXL3 — TabbyAPI recipe

Portable single-NVIDIA-GPU recipe for [Victor's 6.50bpw EXL3 pack](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/tree/c88277ce7f6b90b723f79b5188c0f1b951732099/6.50bpw) using pinned [TabbyAPI](https://github.com/theroyallab/tabbyAPI) and [Victor's K2-Horizon ExLlamaV3 fork](https://github.com/vcruz305/exllamav3). **Validated on one DGX Spark GB10 so far; other supported NVIDIA GPUs are untested here.** This is a recipe, not a model mirror; GitHub and Hugging Face repositories happen to share the same name. Stock upstream ExLlamaV3 does **not** yet register `K2HorizonForCausalLM`.

## Inputs and requirements

| Input | Pin or requirement |
|---|---|
| Runtime | Linux with a supported NVIDIA GPU, working driver, matching CUDA toolkit (`nvcc`), Python 3.12, git, C/C++ build tools, ninja, sufficient free memory/disk |
| TabbyAPI | `theroyallab/tabbyAPI` `f07131cd8fe34e449fe87cdd3a066b52b96d3cac` |
| ExLlamaV3 K2 fork | `vcruz305/exllamav3` `be01a273244ae4238e2919aecdc86bc08d7fb154` (merged K2 PR #10); build from source, not an upstream wheel |
| Quant | `vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3` revision `c88277ce7f6b90b723f79b5188c0f1b951732099`, **only** `6.50bpw/` |
| Learned-bias source | Public `IFM/K2-Horizon-MoVA-36B-A4B` revision `7730b92d1b574e04663b04023d5d6fa83475432f` |

The four quant shards total **31,288,045,811 bytes**; the model card's 31.34 GB describes the entire folder with metadata. This exact quant revision omits **90 learned BF16 routing biases** (45 attention and 45 MoE). A raw quant directory is not a correct runnable model. The fork's [audited extractor](https://github.com/vcruz305/exllamav3/blob/be01a273244ae4238e2919aecdc86bc08d7fb154/tools/extract_k2_routing_bias_overlay.py) recovers a *separate* overlay from the two public pinned model revisions using bounded HTTP ranges and validates its entire SHA-256: `8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c`. This overlay is specific to **6.50bpw at the pinned revision**; never fabricate zero biases, rename `.mlp.None` tensors, or use it for other packs. Model shards stay untouched.

## Install and run

Use a fresh Linux environment. Defaults are Python 3.12 and PyTorch `2.13.0+cu130` from the aarch64/x86_64 CUDA 13 wheel index, **not a GB10-only code path**. If your supported GPU/toolkit needs a different matching CUDA wheel, explicitly set `TORCH_SPEC` and `TORCH_INDEX_URL` before running `install.sh`; record them in any result. No architecture-specific `TORCH_CUDA_ARCH_LIST` is forced. `$WORK_DIR` defaults to `$HOME/k2-horizon-exl3`; use a writable folder with room for the pack and the venv.

```bash
# Optional for a compatible CUDA 12.8 host, for example:
# export TORCH_SPEC='torch==2.13.0+cu128' TORCH_INDEX_URL='https://download.pytorch.org/whl/cu128'
bash scripts/install.sh
bash scripts/download.sh
bash scripts/extract_overlay.sh
"${WORK_DIR:-$HOME/k2-horizon-exl3}/.venv/bin/python" scripts/verify_assets.py \
  --model-dir "${WORK_DIR:-$HOME/k2-horizon-exl3}/models/6.50bpw" \
  --overlay "${WORK_DIR:-$HOME/k2-horizon-exl3}/biases/k2-routing-bias-overlay.safetensors"
bash scripts/serve.sh
```

`install.sh` pins both Git checkouts, installs a CUDA PyTorch wheel, builds the K2 fork against it, and installs **`uvloop` explicitly** (the pinned TabbyAPI can fail at startup with `ModuleNotFoundError: No module named 'uvloop'` without it). Its final import smoke checks Torch CUDA availability, the fork, and `uvloop`. Check that your specific wheel exists and your CUDA toolkit matches. It does not touch another venv. `download.sh` retrieves **only** `6.50bpw/*` from the pinned public HF revision. `extract_overlay.sh` invokes the **fork's** pinned `tools/extract_k2_routing_bias_overlay.py --output-dir ...`; no private overlay URL, HF credential, or whole-shard download is required. It fails closed if the public CDN changes, ignores Range, or changes ETag. Artifacts and manifest stay outside this Git repository.

`serve.sh` validates the local shards, index, overlay SHA and both checkout pins, then patches the *local* TabbyAPI checkout's exact initial-model callsite to pass `routing_bias_overlay` explicitly to `Config.from_directory`. It refuses an unexpected callsite or pin instead of silently serving without learned biases. It writes a local config: loopback `127.0.0.1:5001`, CORS origins `[]`, auth disabled **only because the server is loopback-only**, `model_name: 6.50bpw`, 4096-token context/cache, batch limit 1, no draft or CPU offload. Do not expose or reverse-proxy this unauthenticated benchmark listener. A package update or restored checkout must be re-patched through `serve.sh` after pin verification.

In a second terminal:

```bash
curl --fail --silent http://127.0.0.1:5001/v1/models
curl --fail --silent http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"6.50bpw","messages":[{"role":"user","content":"Reply briefly: what is 2 + 2?"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

Confirm `/v1/models` advertises the exact ID and the response is nonempty; otherwise use the observed model ID rather than assuming it. This is only a smoke, not a quality evaluation. The model card recommends `reasoning_effort: high`, `temperature: 1.0`, `top_p: 0.95` for normal reasoning.

## Measured speed baseline — GB10 only

A **previous**, real Sixcat v0.7.0 unscored speed receipt on one GB10 Spark2 with 6.50bpw/TabbyAPI, **fixed C=1**, strict `temperature=0`, thinking off, seed 1, warmup plus **8/8 successful confirmations per profile**:

| Client-observed metric | Result |
|---|---:|
| Decode effective per-stream p50 (32 prompt words / 512 output max) | **19.28262958139193 tok/s** |
| Balanced aggregate output (256 words / 128 output max) | **17.65341972782038 tok/s** |
| Prefill effective prompt p50 (2048 words / 32 output max) | **1394.911645461034 tok/s** |

The stream exposed no native server-side timing. These metrics are **not universal GPU speeds**, not a concurrency knee, not scored Sixcat capability, and not a new benchmark of these published scripts or of unmerged mixed-K optimization work. The original local receipt contains internal paths and a full chat template, so it is deliberately **not** published here. Other GPU support/performance requires new validation, not extrapolation.

After checking `/v1/models`, reproduce the *same fixed-C1 protocol* (your fresh result may differ):

```bash
git clone https://github.com/vcruz305/sixcat-eval.git "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-eval"
git -C "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-eval" checkout --detach v0.7.0
"${WORK_DIR:-$HOME/k2-horizon-exl3}/.venv/bin/python" -m pip install -e "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-eval"
"${WORK_DIR:-$HOME/k2-horizon-exl3}/.venv/bin/python" -m sixcat speed \
  --base-url http://127.0.0.1:5001/v1 --model 6.50bpw \
  --profile all --concurrency 1 --samples 8 --max-seconds 900 \
  --policy strict --thinking off --out "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-speed.json"
```

The 900-second cap is not an observed runtime. Record actual GPU, CUDA/PyTorch, revision, model ID, overlay digest, policy, successes/failures and client-versus-server metric semantics before publishing comparisons.

## Troubleshooting

- `Unknown architecture K2HorizonForCausalLM`: installed stock ExLlamaV3 or incorrect fork pin; rerun `install.sh` and verify import path/version.
- `Required K2 Horizon selection bias ... missing`: overlay absent or TabbyAPI's local patch not applied; do **not** zero-fill biases. Rerun extractor, verifier and `serve.sh`.
- Extractor fails on Range/ETag/digest: pinned HF object/CDN behavior changed; stop and inspect. Never substitute an unverified overlay.
- Torch/extension compile failure: confirm a matching CUDA-enabled Torch wheel, driver and `nvcc`; do not apply GPU-specific flags without testing that GPU.
- OOM: shared memory/VRAM is finite; reduce cache size in the generated local config and report the change. Keep C=1 for baseline comparison.
- Unauthorized/connection error: verify the listener and `127.0.0.1` port locally. Do not open it on the network to “fix” access.

## Credits and license

[IFM's original K2-Horizon model](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) is Apache-2.0. EXL3 and ExLlamaV3 are by [Turboderp](https://github.com/turboderp-org/exllamav3) (MIT). [TabbyAPI](https://github.com/theroyallab/tabbyAPI) is AGPL-3.0. Their licenses remain separate; this repository contains original recipe scripts/docs only under [MIT](LICENSE), no weights or vendored runtimes. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Independent community recipe, not an endorsement by IFM, NVIDIA or runtime maintainers.
