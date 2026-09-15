#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def optional_module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    output = ROOT_DIR / ".artifacts" / "environment.json"
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": freeze,
        "torch": optional_module_version("torch"),
        "transformers": optional_module_version("transformers"),
        "datasets": optional_module_version("datasets"),
    }
    try:
        import torch

        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_version"] = torch.version.cuda
        payload["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Environment written to ignored local artifact: {output}")


if __name__ == "__main__":
    main()
