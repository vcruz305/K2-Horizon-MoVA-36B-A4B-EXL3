"""Fail-closed validation for six pinned K2 Horizon EXL3 packs and their BF16 overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

EXPECTED_OVERLAY_SHA256 = "8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c"
MODEL_REV = "db645b888e6e05e89e7922230e0c1d526f87615c"
PACK_SHARDS = {
    "8.00": {
        "model-00001-of-00005.safetensors": 7944229765,
        "model-00002-of-00005.safetensors": 8002430284,
        "model-00003-of-00005.safetensors": 7989159179,
        "model-00004-of-00005.safetensors": 7993582918,
        "model-00005-of-00005.safetensors": 6089603087,
        "model-routing-bias.safetensors": 39784,
    },
    "6.50": {
        "model-00001-of-00004.safetensors": 7941611589,
        "model-00002-of-00004.safetensors": 7992603956,
        "model-00003-of-00004.safetensors": 8348087304,
        "model-00004-of-00004.safetensors": 7005761354,
    },
    "5.00": {
        "model-00001-of-00003.safetensors": 7828889640,
        "model-00002-of-00003.safetensors": 8481780968,
        "model-00003-of-00003.safetensors": 8198023024,
    },
    "4.00": {
        "model-00001-of-00003.safetensors": 8378769476,
        "model-00002-of-00003.safetensors": 8490277873,
        "model-00003-of-00003.safetensors": 3120022441,
    },
    "2.50": {
        "model-00001-of-00002.safetensors": 8468875257,
        "model-00002-of-00002.safetensors": 4740701376,
    },
    "2.00": {
        "model-00001-of-00002.safetensors": 8441052506,
        "model-00002-of-00002.safetensors": 2508719654,
    },
}
# Backward-compatible test alias for the original default pack.
SHARDS = PACK_SHARDS["6.50"]
ARCH = "K2HorizonForCausalLM"


def verify_overlay(path: Path) -> None:
    if path.name != "k2-routing-bias-overlay.safetensors" or not path.is_file():
        raise ValueError(f"Missing expected routing-bias overlay: {path}")
    raw = path.read_bytes()  # 90 tiny BF16 vectors, never a model shard.
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_OVERLAY_SHA256:
        raise ValueError(f"Overlay SHA-256 mismatch: {digest}")
    if len(raw) < 8:
        raise ValueError("Invalid safetensors header")
    length = struct.unpack("<Q", raw[:8])[0]
    if length < 1 or length > 1_000_000 or length + 8 > len(raw):
        raise ValueError("Invalid safetensors header length")
    header = json.loads(raw[8:8 + length])
    expected = {
        f"model.layers.{layer}.{part}.bias": count
        for layer in range(3, 48)
        for part, count in (("self_attn.v_router", 64), ("mlp.gate", 100))
    }
    if set(header) != set(expected):
        raise ValueError("Overlay must contain exactly 90 named learned biases")
    for key, count in expected.items():
        info = header[key]
        start, end = info["data_offsets"]
        if (info["dtype"] != "BF16" or info["shape"] != [count]
                or start < 0 or end - start != 2 * count or 8 + length + end > len(raw)):
            raise ValueError(f"Invalid bias tensor: {key}")
    print("Verified 90 BF16 routing biases and pinned overlay SHA-256")


def verify_model(path: Path) -> None:
    pack = path.name.removesuffix("bpw") if path.name.endswith("bpw") else ""
    shards = PACK_SHARDS.get(pack)
    if not path.is_dir() or shards is None:
        raise ValueError(f"Expected one of the pinned quant folders: {', '.join(PACK_SHARDS)}")
    for name in ("config.json", "model.safetensors.index.json", "quantization_config.json", "tokenizer.json", "chat_template.jinja"):
        if not (path / name).is_file():
            raise ValueError(f"Missing model metadata: {name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != [ARCH]:
        raise ValueError(f"Wrong model architecture: {config.get('architectures')}")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    referenced = set(index["weight_map"].values())
    if not index["weight_map"] or referenced != set(shards):
        raise ValueError("Wrong/incomplete safetensors index")
    for name, size in shards.items():
        shard = path / name
        if not shard.is_file() or shard.stat().st_size != size:
            raise ValueError(f"Missing/truncated/incorrect shard: {name}")
    expected_aliases = {
        f"model.layers.{layer}.{part}.e_score_correction_bias"
        for layer in range(3, 48)
        for part in ("self_attn.v_router", "mlp.gate")
    }
    if not expected_aliases <= set(index["weight_map"]):
        raise ValueError("Incomplete routing-bias index for the pinned pack")
    print(f"Verified {pack}bpw K2 config, {len(shards)} index-referenced files, byte sizes and 90 bias aliases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()
    verify_overlay(args.overlay)
    if args.model_dir is not None:
        verify_model(args.model_dir)


if __name__ == "__main__":
    main()
