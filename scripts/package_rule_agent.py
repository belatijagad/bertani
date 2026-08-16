#!/usr/bin/env python3
"""Build a deterministic Kaggle submission archive for the rule-based agent."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "bertani"
RULE_VERSIONS = ROOT / "src" / "bertani_rules"
DEFAULT_OUTPUT = ROOT / "dist" / "rule_based_submission.tar.gz"
MODULES = (
    "vec_env.py",
    "opening.py",
    "market.py",
    "tasks.py",
    "rule_based.py",
    "kaggle_agent.py",
)
MAIN = b'''"""Bertani versioned rule-based Kaggriculture submission."""
from bertani.kaggle_agent import make_agent
from rule_version import build_policy

agent = make_agent(build_policy)

__all__ = ["agent"]
'''
INIT = b'"""Portable Bertani rule-agent abstractions."""\n'


def archive_members(version: str) -> dict[str, bytes]:
    """Collect the root entry point and its pure-Python dependencies."""
    if re.fullmatch(r"[a-z][a-z0-9_]*", version) is None:
        raise ValueError("version must contain lowercase letters, digits, or underscores")
    version_path = RULE_VERSIONS / f"{version}.py"
    if not version_path.is_file():
        available = ", ".join(
            path.stem
            for path in sorted(RULE_VERSIONS.glob("*.py"))
            if path.name != "__init__.py"
        )
        raise FileNotFoundError(
            f"unknown rule version {version!r}; available versions: {available}"
        )
    version_payload = version_path.read_bytes()
    compile(version_payload, str(version_path), "exec")
    members = {
        "main.py": MAIN,
        "bertani/__init__.py": INIT,
        "rule_version.py": version_payload,
    }
    for name in MODULES:
        path = SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(f"required rule-agent module is missing: {path}")
        payload = path.read_bytes()
        compile(payload, str(path), "exec")
        members[f"bertani/{name}"] = payload
    manifest = {
        "format": 1,
        "entrypoint": "main.py:agent",
        "rule_version": version,
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(members.items())
        },
    }
    members["MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    return members


def build_archive(output: Path, version: str = "v1") -> str:
    """Write an order-stable, timestamp-free tar.gz and return its SHA-256."""
    members = archive_members(version)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(members):
                    payload = members[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))

    payload = output.read_bytes()
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        if "main.py" not in names or "bertani/kaggle_agent.py" not in names:
            raise RuntimeError("submission archive is missing its entry point or adapter")
        for member in archive.getmembers():
            if not member.isfile():
                raise RuntimeError(f"unexpected non-file archive member: {member.name}")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="v1",
        help="rule version from src/bertani_rules (default: v1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archive destination (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    digest = build_archive(output, args.version)
    print(f"built {output}")
    print(f"rule version {args.version}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
