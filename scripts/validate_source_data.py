#!/usr/bin/env python3
"""Validate Wiki inputs and, when requested, the domain-shift source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_unique(root: Path, basename: str) -> Path:
    matches = sorted(path for path in root.rglob(basename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Required source file {basename!r} not found under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {basename!r} files found. Pass a narrower source directory.")
    return matches[0]


def validate_dataset(name: str, source_root: Path, config: dict, allow_different_snapshot: bool) -> None:
    mismatches = []
    for split_name, file_spec in config["source_files"].items():
        path = find_unique(source_root, file_spec["basename"])
        actual = sha256_path(path)
        expected = file_spec["sha256"]
        if actual != expected:
            mismatches.append((split_name, path, expected, actual))
        else:
            print(f"{name}/{split_name}: checksum verified ({path.name})")
    if mismatches and not allow_different_snapshot:
        details = "\n".join(
            f"  {split}: {path.name}\n    expected {expected}\n    actual   {actual}"
            for split, path, expected, actual in mismatches
        )
        raise RuntimeError(
            f"{name} does not match the thesis source snapshot:\n{details}\n"
            "Pass --allow-different-snapshot only for a documented, non-equivalent replication."
        )
    for split, path, _, _ in mismatches:
        print(f"WARNING: {name}/{split} differs from the thesis snapshot ({path.name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-source", type=Path)
    parser.add_argument("--twitter-source", type=Path, help="Researcher-supplied source for domain-shift reproduction only")
    parser.add_argument("--allow-different-snapshot", action="store_true")
    args = parser.parse_args()
    if args.wiki_source is None and args.twitter_source is None:
        raise SystemExit("Pass --wiki-source, --twitter-source, or both.")
    payload = json.loads((ROOT_DIR / "configs" / "data_sources.json").read_text(encoding="utf-8"))
    if args.wiki_source is not None:
        validate_dataset(
            "wiki_ner", args.wiki_source.resolve(), payload["datasets"]["wiki_ner"], args.allow_different_snapshot
        )
    if args.twitter_source is not None:
        validate_dataset(
            "twitter_ner",
            args.twitter_source.resolve(),
            payload["datasets"]["twitter_ner"],
            args.allow_different_snapshot,
        )


if __name__ == "__main__":
    main()
