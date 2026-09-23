"""Offline fail-closed tests; no GPU, secrets, or model download required."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import patch_tabby
import verify_assets


class RecipeTests(unittest.TestCase):
    def test_patch_exact_callsite_and_idempotence(self):
        source = "import asyncio\nimport pathlib\n" + patch_tabby.ORIGINAL + "\n"
        changed = patch_tabby.patch_text(source)
        self.assertIn("import os\n", changed)
        self.assertIn("K2_ROUTING_BIAS_OVERLAY", changed)
        self.assertEqual(patch_tabby.patch_text(changed), changed)

    def test_patch_rejects_missing_or_duplicate_callsite(self):
        for source in ("import asyncio\nimport pathlib\n", "import asyncio\nimport pathlib\n" + patch_tabby.ORIGINAL * 2):
            with self.assertRaises(ValueError):
                patch_tabby.patch_text(source)

    def test_missing_overlay_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Missing expected"):
                verify_assets.verify_overlay(Path(folder) / "k2-routing-bias-overlay.safetensors")

    def test_wrong_overlay_digest_fails_even_if_header_looks_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "k2-routing-bias-overlay.safetensors"
            header = b'{"fake":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_assets.verify_overlay(path)

    def test_other_pack_loads_when_its_index_and_shards_match(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "4.00bpw"
            model.mkdir()
            for name, content in (("config.json", json.dumps({"architectures": [verify_assets.ARCH]})),
                                  ("quantization_config.json", "{}"),
                                  ("tokenizer.json", "{}"),
                                  ("chat_template.jinja", "template")):
                (model / name).write_text(content)
            name = "model-00001-of-00003.safetensors"
            biases = {
                f"model.layers.{layer}.{part}.e_score_correction_bias": name
                for layer in range(3, 48)
                for part in ("self_attn.v_router", "mlp.gate")
            }
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": name, **biases}}))
            (model / name).write_bytes(b"x")
            with patch.dict(verify_assets.PACK_SHARDS, {"4.00": {name: 1}}):
                verify_assets.verify_model(model)

    def test_missing_pack_bias_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "4.00bpw"
            model.mkdir()
            for name, content in (("config.json", json.dumps({"architectures": [verify_assets.ARCH]})),
                                  ("quantization_config.json", "{}"),
                                  ("tokenizer.json", "{}"),
                                  ("chat_template.jinja", "template")):
                (model / name).write_text(content)
            name = "model-00001-of-00003.safetensors"
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": name}}))
            (model / name).write_bytes(b"x")
            with patch.dict(verify_assets.PACK_SHARDS, {"4.00": {name: 1}}):
                with self.assertRaisesRegex(ValueError, "routing-bias index"):
                    verify_assets.verify_model(model)

    def test_download_rejects_unlisted_pack_before_any_network_call(self):
        script = Path(__file__).with_name("download.sh")
        with tempfile.TemporaryDirectory() as folder:
            env = {**os.environ, "WORK_DIR": folder}
            result = subprocess.run(["bash", "-c", "PACK='../wrong' bash scripts/download.sh"],
                                    cwd=script.parent.parent, capture_output=True, text=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported PACK", result.stderr)

    def test_serve_rejects_unlisted_pack_before_model_io(self):
        script = Path(__file__).with_name("serve.sh")
        with tempfile.TemporaryDirectory() as folder:
            env = {**os.environ, "WORK_DIR": folder}
            result = subprocess.run(["bash", "-c", "PACK='../wrong' bash scripts/serve.sh"],
                                    cwd=script.parent.parent, capture_output=True, text=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported PACK", result.stderr)

    def test_model_index_and_shard_sizes_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "6.50bpw"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"architectures": [verify_assets.ARCH]}))
            (model / "quantization_config.json").write_text("{}")
            (model / "tokenizer.json").write_text("{}")
            (model / "chat_template.jinja").write_text("template")
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "wrong.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "index"):
                verify_assets.verify_model(model)
            weight_map = {str(i): name for i, name in enumerate(verify_assets.SHARDS)}
            weight_map.update({
                f"model.layers.{layer}.{part}.e_score_correction_bias": next(iter(verify_assets.SHARDS))
                for layer in range(3, 48)
                for part in ("self_attn.v_router", "mlp.gate")
            })
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            for name in verify_assets.SHARDS:
                (model / name).write_bytes(b"a")
            with self.assertRaisesRegex(ValueError, "shard"):
                verify_assets.verify_model(model)
            with patch.dict(verify_assets.SHARDS, {name: 1 for name in verify_assets.SHARDS}, clear=True):
                verify_assets.verify_model(model)


if __name__ == "__main__":
    unittest.main()
