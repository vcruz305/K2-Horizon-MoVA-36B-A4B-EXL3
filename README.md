# K2-Horizon-MoVA-36B-A4B · EXL3

Run any of the six [K2-Horizon-MoVA-36B-A4B EXL3 packs](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3) on a compatible Linux NVIDIA GPU. Select a bitrate with `PACK`; the scripts download only that folder and serve it through pinned [TabbyAPI](https://github.com/theroyallab/tabbyAPI) and the [K2-capable ExLlamaV3 fork](https://github.com/vcruz305/exllamav3). This GitHub repository is a recipe, **not** a weight mirror. Measured speeds so far: 8.00bpw on one RTX PRO 6000 Blackwell (development runtime, not these scripts) and the older 6.50bpw revision on one GB10; other packs and GPUs require their own runtime checks. Stock upstream ExLlamaV3 does not yet register `K2HorizonForCausalLM`.

## Choose a pack

The following table is copied from the [Hugging Face model README](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/blob/main/README.md). Sizes are whole folders, not just model shards; the smallest-card column is the model-card estimate, **not** a validated TabbyAPI minimum for this recipe.

| bpw | size | smallest card | top-1 vs BF16 | mean KLD | p99 KLD | download |
| --- | --- | --- | --- | --- | --- | --- |
| **8.00** | 38.07 GB | 96 / 48 GB | 96.43% (9,874 / 10,240) | 0.0047 | 0.042 | [8.00bpw](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/tree/main/8.00bpw) |
| **6.50** | 31.34 GB | 48 GB | 90.68% (9,286 / 10,240) | 0.0335 | 0.336 | [6.50bpw](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/tree/main/6.50bpw) |
| **5.00** | 24.56 GB | 32 GB | 85.83% (8,789 / 10,240) | 0.0923 | 0.826 | [5.00bpw](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/tree/main/5.00bpw) |
| **4.00** | 20.05 GB | 24 GB | 84.81% (8,685 / 10,240) | 0.1082 | 0.909 | [4.00bpw](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/tree/main/4.00bpw) |
| **2.50** | 13.27 GB | 16 GB | 83.71% (8,572 / 10,240) | 0.1326 | 1.153 | [2.50bpw](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/tree/main/2.50bpw) |
| **2.00** | 11.01 GB | 8 GB + offload | 81.66% (8,362 / 10,240) | 0.1604 | 1.322 | [2.00bpw](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/tree/main/2.00bpw) |

**Top-1** is next-token agreement with the original BF16 model on 10,240 held-out positions. **KLD** is the mean/99th-percentile KL(reference ‖ pack) on those positions; lower is closer. These are quantization-fidelity measures, not task-accuracy scores. The 2.00bpw estimate for an 8 GB card **requires expert offload**, which this recipe does not enable by default; choose adequate memory or configure and validate offload separately.

## Inputs and requirements

| Input | Pin or requirement |
|---|---|
| Runtime | Linux with a supported NVIDIA GPU, working driver, matching CUDA toolkit (`nvcc`), Python 3.12, git, C/C++ build tools, ninja, sufficient free memory and disk |
| TabbyAPI | `theroyallab/tabbyAPI` `f07131cd8fe34e449fe87cdd3a066b52b96d3cac` |
| ExLlamaV3 K2 fork | `vcruz305/exllamav3` `be01a273244ae4238e2919aecdc86bc08d7fb154` (merged K2 PR #10); build from source, not an upstream wheel |
| All six HF folders | `vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3` revision `db645b888e6e05e89e7922230e0c1d526f87615c`; select exactly one via `PACK` |
| Canonical learned-bias source | Public `IFM/K2-Horizon-MoVA-36B-A4B` revision `7730b92d1b574e04663b04023d5d6fa83475432f` |

**Routing biases matter.** All six pinned pack indexes contain 90 learned correction-bias aliases, but the pinned K2 runtime expects the 45 MoVA and 45 MoE selection biases under their original `.bias` keys. `extract_overlay.sh` runs the fork's audited extractor to produce a *separate* canonical BF16 overlay from public pinned sources; `verify_assets.py` checks its entire SHA-256 (`8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c`), 90 keys, shapes and dtype. The extractor cross-checks against the older pinned 6.50bpw revision used to establish the canonical overlay; that historical cross-check is **not** the selected pack download revision. Apply the same validated original-model biases regardless of selected bitrate. Never fabricate zero biases, rename tensors inside weight shards, or mix unverified overlays. The 8.00 folder also has an index-referenced `model-routing-bias.safetensors` sidecar; keep it with its pack. Shard byte-size checks do not constitute full shard hashes.

## Install and run

Use a fresh Linux environment. Pick **one** of `8.00`, `6.50`, `5.00`, `4.00`, `2.50`, or `2.00` before download and serving; the following is an example, not a preferred pack. `$WORK_DIR` defaults to `$HOME/k2-horizon-exl3`; use a writable location with room for the selected folder and the venv.

```bash
export PACK=4.00
# Optional for a compatible CUDA 12.8 host, for example:
# export TORCH_SPEC='torch==2.13.0+cu128' TORCH_INDEX_URL='https://download.pytorch.org/whl/cu128'
bash scripts/install.sh
bash scripts/download.sh
bash scripts/extract_overlay.sh
"${WORK_DIR:-$HOME/k2-horizon-exl3}/.venv/bin/python" scripts/verify_assets.py \
  --model-dir "${WORK_DIR:-$HOME/k2-horizon-exl3}/models/${PACK}bpw" \
  --overlay "${WORK_DIR:-$HOME/k2-horizon-exl3}/biases/k2-routing-bias-overlay.safetensors"
bash scripts/serve.sh
```

`install.sh` pins both Git checkouts, installs Python 3.12 PyTorch `2.13.0+cu130` by default from the aarch64/x86_64 CUDA 13 wheel index, builds the K2 fork against it, and installs `uvloop` explicitly. Choose a compatible wheel/driver/toolkit for your GPU using `TORCH_SPEC` and `TORCH_INDEX_URL`; no GPU architecture is hard-coded. `download.sh` fetches **only** `${PACK}bpw/*` from the six-pack pin. `extract_overlay.sh` uses public bounded HTTP ranges, refuses changed ETags/digests and writes outside the Git repository. `serve.sh` checks the selected pack's config, 90 bias-index aliases, exact index-referenced shard names/byte sizes, overlay digest and both runtime checkout pins, then narrowly patches the local TabbyAPI callsite to pass the overlay. It writes a local loopback-only config (`127.0.0.1:5001`, model ID `${PACK}bpw`, 4096-token context/cache, batch limit 1, no draft or offload). **Do not expose or reverse-proxy the unauthenticated benchmark listener.**

In a second terminal, with the same `PACK` value:

```bash
curl --fail --silent http://127.0.0.1:5001/v1/models
curl --fail --silent http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${PACK}bpw\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply briefly: what is 2 + 2?\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}"
```

Confirm `/v1/models` advertises your pack's ID and the completion is nonempty; this is a smoke, not a quality evaluation. For normal reasoning, IFM recommends `reasoning_effort: high`, `temperature: 1.0`, `top_p: 0.95`. The pack must fit your actual available GPU memory; the table is not a promise that every card can serve every pack.

## Measured speed

Each result is tied to one GPU, pack, runtime and set of server settings. Neither predicts another GPU, another pack, or this recipe's TabbyAPI scripts.

### 8.00bpw on one RTX PRO 6000 Blackwell (96 GB)

| Setting | Value |
|---|---|
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, driver 580.178.04 |
| Software | PyTorch 2.13.0+cu129, CUDA 12.9 |
| Pack | `8.00bpw` at revision `db645b888e6e05e89e7922230e0c1d526f87615c` (the pin above); all 18 files hash-checked against the Hub |
| Runtime | Development ExLlamaV3 build with K2 serving changes that are not yet in the pinned fork, behind a native OpenAI-compatible `/v1` server. **Not** this recipe's TabbyAPI scripts, which have not been speed-tested on this GPU. |
| Server | 65,536-token cache, up to 16 active requests, no draft model, no offload; 48,295 MiB GPU memory in use after load |
| Benchmark | [SixCat](https://github.com/vcruz305/sixcat-eval) v0.7.0 `sixcat speed`, default suite: decode, balanced and prefill profiles. Each profile runs a concurrency curve (C = 1, 2, 4, 8 with 4, 4, 8, 16 requests), then a 32-request confirmation at the selected concurrency. One warmup request precedes each curve and each confirmation. |
| Sampling | SixCat `strict` policy: `temperature=0`, thinking off, seed 1. Every request generated exactly its `max_tokens`. |
| Run | 2026-09-22, 8:32–8:51 PM PDT; 192/192 measured requests succeeded |

SixCat found no server-side timing on this route, so every number is client-observed over streaming. **Aggregate** is total output tokens ÷ wall time for the level, including prefill and queueing. **Per-stream decode** is one request's tokens after its first token ÷ the time after its first token. **TTFT** includes any wait for a free slot.

SixCat summary:

| Metric | Result |
|---|---:|
| Decode, single stream (C=1): p50 / max | **30.0 / 30.1 tok/s** |
| Prefill, single stream (C=1, ≈2,470-token prompt): p50 | **2,763 tok/s** |
| Balanced serving, aggregate output at C=8 | **46.0 tok/s** |
| Time to first token, balanced: C=1 p50 / C=8 p50 / C=8 p95 | 686 / 4,730 / 4,740 ms |
| Highest usable concurrency tested | 8 (100% success at every level) |

Concurrency curve (prompt → output tokens per request):

| Profile | C | OK | Aggregate output tok/s | Per-stream decode p50 tok/s | TTFT p50 |
|---|---:|---:|---:|---:|---:|
| Decode (≈120 → 512) | 1 | 4/4 | 29.06 | 29.97 | 580 ms |
| | 2 | 4/4 | 21.34 | 10.86 | 906 ms |
| | 4 | 8/8 | 30.04 | 7.70 | 1,783 ms |
| | 8 | 16/16 | 42.85 | 5.57 | 4,110 ms |
| Balanced (≈380 → 128) | 1 | 4/4 | 26.16 | 30.23 | 686 ms |
| | 2 | 4/4 | 20.50 | 11.13 | 1,074 ms |
| | 4 | 8/8 | 30.75 | 8.81 | 2,728 ms |
| | 8 | 16/16 | 45.24 | 7.19 | 5,342 ms |
| Prefill (≈2,470 → 32) | 1 | 4/4 | 16.63 | 30.14 | 894 ms |
| | 2 | 4/4 | 14.05 | 9.96 | 1,377 ms |
| | 4 | 8/8 | 20.00 | 10.85 | 3,549 ms |
| | 8 | 16/16 | 24.12 | 6.96 | 6,138 ms |

Confirmation at C=8, 32 requests per profile:

| Profile | OK | Aggregate output tok/s | Per-stream decode p50 / p95 tok/s | TTFT p50 / p95 | End-to-end p50 |
|---|---:|---:|---:|---:|---:|
| Decode | 32/32 | 48.67 | 6.24 / 7.39 | 4,543 / 4,598 ms | 84.8 s |
| Balanced | 32/32 | 46.00 | 7.25 / 7.44 | 4,730 / 4,740 ms | 22.2 s |
| Prefill | 32/32 | 24.17 | 6.98 / 6.99 | 6,150 / 6,161 ms | 10.6 s |

Concurrency scales poorly in this build: C=2 gives less total output than C=1, C=4 is only 3–20% above C=1, and C=8 reaches about 1.5–1.8× the C=1 total while each stream slows to 5.6–7.3 tok/s. p99 is not reported because no level has 100 or more requests.

Command used. Everything not shown is a v0.7.0 default (`--profile all --candidates 1,2,4,8 --samples 32 --policy strict --thinking off --seed 1`). The time caps are raised because at about 30 tok/s the default 600-second suite budget runs out during the decode profile, and v0.7.0 then writes no results file.

```bash
python -m sixcat speed \
  --base-url http://127.0.0.1:<port>/v1 --model <served-model-id> \
  --max-seconds 7200 --curve-seconds 1500 \
  --out sixcat-speed-8.00.json
```

### Older 6.50bpw revision on one GB10 (historical)

A **previous** Sixcat v0.7.0 unscored speed receipt on one GB10 Spark2 used the **older** `6.50bpw` revision `c88277ce7f6b90b723f79b5188c0f1b951732099`, TabbyAPI, fixed C=1, strict `temperature=0`, thinking off, seed 1, warmup and 8/8 successful confirmations per profile:

| Client-observed metric | Result |
|---|---:|
| Decode effective per-stream p50 (32 prompt words / 512 output max) | **19.28262958139193 tok/s** |
| Balanced aggregate output (256 words / 128 output max) | **17.65341972782038 tok/s** |
| Prefill effective prompt p50 (2048 words / 32 output max) | **1394.911645461034 tok/s** |

These are **not** universal GPU speeds, measurements of the other five packs, or benchmarks of the newer six-pack HF revision and these published scripts. The stream exposed no native server-side timing. The original local receipt contains internal paths and a full chat template, so it is not published here.

### Measure your own pack

For a *new* result for your selected pack, run the same fixed-C1 Sixcat protocol after confirming the model ID:

```bash
git clone https://github.com/vcruz305/sixcat-eval.git "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-eval"
git -C "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-eval" checkout --detach v0.7.0
"${WORK_DIR:-$HOME/k2-horizon-exl3}/.venv/bin/python" -m pip install -e "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-eval"
"${WORK_DIR:-$HOME/k2-horizon-exl3}/.venv/bin/python" -m sixcat speed \
  --base-url http://127.0.0.1:5001/v1 --model "${PACK}bpw" \
  --profile all --concurrency 1 --samples 8 --max-seconds 900 \
  --policy strict --thinking off --out "${WORK_DIR:-$HOME/k2-horizon-exl3}/sixcat-speed-${PACK}.json"
```

The recipe's server runs one request at a time, so keep `--concurrency 1` against it; the concurrency curve above needs a server that batches. The 900-second cap is not an observed runtime. Record actual GPU, CUDA/PyTorch, pack/revision, overlay digest, policy, successes/failures and client-versus-server metric semantics before making comparisons.

## Troubleshooting

- `Unknown architecture K2HorizonForCausalLM`: installed stock ExLlamaV3 or incorrect fork pin; rerun `install.sh` and verify the import path.
- `Required K2 Horizon selection bias ... missing`: original-key overlay absent or TabbyAPI's local patch not applied; do **not** zero-fill. Rerun extractor, verifier and `serve.sh`.
- Extractor fails on Range/ETag/digest: a pinned public object/CDN changed; stop and inspect. Do not substitute an unverified overlay.
- Torch/extension compile failure: confirm matching CUDA-enabled Torch wheel, driver and `nvcc`; do not apply GPU-specific flags without testing that GPU.
- OOM: ensure your card can hold the selected pack plus KV cache; the 2.00bpw 8 GB case needs separately configured expert offload. Reduce cache size in generated local config if needed and report the change.
- Unauthorized/connection error: verify `127.0.0.1:5001` locally. Do not open it on the network to “fix” access.

## Credits and license

[IFM's original K2-Horizon model](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) is Apache-2.0. EXL3 and ExLlamaV3 are by [Turboderp](https://github.com/turboderp-org/exllamav3) (MIT). [TabbyAPI](https://github.com/theroyallab/tabbyAPI) is AGPL-3.0. Their licenses remain separate; this repository contains original recipe scripts/docs only under [MIT](LICENSE), no weights or vendored runtimes. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Independent community recipe, not an endorsement by IFM, NVIDIA or runtime maintainers.
