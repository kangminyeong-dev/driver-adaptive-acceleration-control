#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return "not found"
    return completed.stdout.strip() or "no output"


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, getattr(module, "__version__", "version unknown")


def main() -> int:
    print("[Jetson runtime check]")
    print(f"- root: {ROOT}")
    print(f"- python: {sys.version.split()[0]} ({sys.executable})")
    print(f"- machine: {platform.machine()}")
    print(f"- l4t: {command_output(['dpkg-query', '-W', 'nvidia-l4t-core'])}")
    print(f"- nvcc: {command_output(['nvcc', '--version']).splitlines()[-1]}")

    required = ["PIL", "torch", "torchvision"]
    missing = []
    for module_name in required:
        ok, detail = import_status(module_name)
        label = "Pillow" if module_name == "PIL" else module_name
        print(f"- {label}: {'ok' if ok else 'missing'} ({detail})")
        if not ok:
            missing.append(label)

    weights_path = ROOT / "weights" / "best.pt"
    config_path = ROOT / "config" / "runtime_config.json"
    samples_path = ROOT / "samples"
    print(f"- weights/best.pt: {'ok' if weights_path.is_file() else 'missing'}")
    print(f"- config/runtime_config.json: {'ok' if config_path.is_file() else 'missing'}")
    print(f"- samples: {'ok' if samples_path.is_dir() else 'missing'}")

    if missing:
        print()
        print("Missing packages:", ", ".join(missing))
        print("Install Jetson-compatible torch/torchvision before running inference.")
        return 1

    import torch

    print(f"- torch cuda available: {torch.cuda.is_available()}")
    print(f"- torch cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"- cuda device 0: {torch.cuda.get_device_name(0)}")

    print()
    print("Ready. Try: python3 src/infer_image.py samples/0000.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
