"""Patch exactly one pinned local TabbyAPI callsite to pass the verified overlay."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

TABBY_PIN = "f07131cd8fe34e449fe87cdd3a066b52b96d3cac"
ORIGINAL = "self.config = Config.from_directory(str(model_directory.resolve()))"
PATCHED = (
    "self.config = Config.from_directory("
    "str(model_directory.resolve()), "
    "routing_bias_overlay=str(pathlib.Path(os.environ[\"K2_ROUTING_BIAS_OVERLAY\"]).resolve()))"
)


def patch_text(text: str) -> str:
    if PATCHED in text:
        if text.count(PATCHED) != 1 or "import os\n" not in text or ORIGINAL in text:
            raise ValueError("Unexpected prepatched TabbyAPI source")
        return text
    if text.count(ORIGINAL) != 1 or "import pathlib\n" not in text or not text.startswith("import asyncio\n"):
        raise ValueError("Unexpected TabbyAPI callsite; refusing patch")
    return text.replace("import asyncio\n", "import asyncio\nimport os\n", 1).replace(ORIGINAL, PATCHED, 1)


def patch_checkout(root: Path) -> None:
    pin = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if pin != TABBY_PIN:
        raise ValueError(f"Refusing TabbyAPI commit {pin}; expected {TABBY_PIN}")
    path = root / "backends" / "exllamav3" / "model.py"
    source = path.read_text(encoding="utf-8")
    changed = patch_text(source)
    if changed != source:
        path.write_text(changed, encoding="utf-8")
    print("Pinned TabbyAPI initial model passes K2_ROUTING_BIAS_OVERLAY explicitly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tabby_checkout", type=Path)
    patch_checkout(parser.parse_args().tabby_checkout)


if __name__ == "__main__":
    main()
