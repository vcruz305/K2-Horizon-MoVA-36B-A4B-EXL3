"""Fail-closed validation for this single pinned 6.50bpw pack and BF16 overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

EXPECTED_OVERLAY_SHA256 = "8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c"
SHARDS = {
    "model-00001-of-00004.safetensors": 7941608324,
    "model-00002-of-00004.safetensors": 7992599868,
    "model-00003-of-00004.safetensors": 8348082807,
    "model-00004-of-00004.safetensors": 7005754812,
}
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
    if not path.is_dir() or path.name != "6.50bpw":
        raise ValueError("Expected the 6.50bpw quant folder")
    for name in ("config.json", "model.safetensors.index.json", "quantization_config.json", "tokenizer.json", "chat_template.jinja"):
        if not (path / name).is_file():
            raise ValueError(f"Missing model metadata: {name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != [ARCH]:
        raise ValueError(f"Wrong model architecture: {config.get('architectures')}")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    referenced = set(index["weight_map"].values())
    if not index["weight_map"] or referenced != set(SHARDS):
        raise ValueError("Wrong/incomplete four-shard safetensors index")
    for name, size in SHARDS.items():
        shard = path / name
        if not shard.is_file() or shard.stat().st_size != size:
            raise ValueError(f"Missing/truncated/incorrect shard: {name}")
    print("Verified K2 config, four index-referenced shards and byte sizes")


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
