#!/usr/bin/env python3
"""Build a Kaggle archive for the Rust-backed rule agent."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import platform
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "bertani"
RULE_SOURCE = ROOT / "src" / "bertani_rules" / "agent.py"
DEFAULT_OUTPUT = ROOT / "dist" / "rule_based_submission.tar.gz"

MODULES = (
    "vec_env.py",
    "opening.py",
    "market.py",
    "tasks.py",
    "rule_based.py",
    "kaggle_agent.py",
)

MAIN = b'''"""Bertani rule-based Kaggriculture submission."""
from bertani.kaggle_agent import make_agent
from rules import build_policy

agent = make_agent(build_policy)
__all__ = ["agent"]
'''

INIT = b'"""Portable Bertani rule-agent abstractions."""\n'


def _native_extension() -> Path:
    """Locate the already-built ``bertani._rust`` extension."""

    spec = importlib.util.find_spec("bertani._rust")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "bertani._rust is not built. Run "
            "`uv run maturin develop --release` before packaging."
        )

    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"native extension is missing: {path}")

    if not any(
        path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise RuntimeError(
            f"bertani._rust did not resolve to a CPython extension: {path}"
        )
    return path


def archive_members() -> dict[str, bytes]:
    """Collect the Python agent and its native route solver."""

    if not RULE_SOURCE.is_file():
        raise FileNotFoundError(f"rule strategy is missing: {RULE_SOURCE}")

    rule_payload = RULE_SOURCE.read_bytes()
    compile(rule_payload, str(RULE_SOURCE), "exec")

    native = _native_extension()
    native_member = f"bertani/{native.name}"

    members: dict[str, bytes] = {
        "main.py": MAIN,
        "bertani/__init__.py": INIT,
        "rules.py": rule_payload,
        native_member: native.read_bytes(),
    }

    for name in MODULES:
        path = SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(
                f"required rule-agent module is missing: {path}"
            )
        payload = path.read_bytes()
        compile(payload, str(path), "exec")
        members[f"bertani/{name}"] = payload

    manifest = {
        "format": 2,
        "entrypoint": "main.py:agent",
        "native_extension": native_member,
        "build_python": platform.python_version(),
        "build_cache_tag": sys.implementation.cache_tag,
        "build_platform": platform.platform(),
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(members.items())
        },
    }
    members["MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    return members


def build_archive(output: Path) -> str:
    """Write an order-stable, timestamp-free tar.gz and return its SHA-256."""

    members = archive_members()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            mtime=0,
        ) as zipped:
            with tarfile.open(
                fileobj=zipped,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for name in sorted(members):
                    payload = members[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o755 if name.endswith(".so") else 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))

    payload = output.read_bytes()
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        required = {"main.py", "bertani/kaggle_agent.py", "MANIFEST.json"}
        if not required.issubset(names):
            missing = sorted(required.difference(names))
            raise RuntimeError(
                f"submission archive is missing required files: {missing}"
            )
        if not any(
            name.startswith("bertani/_rust")
            and any(
                name.endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            )
            for name in names
        ):
            raise RuntimeError(
                "submission archive is missing the native Rust extension"
            )
        for member in archive.getmembers():
            if not member.isfile():
                raise RuntimeError(
                    f"unexpected non-file archive member: {member.name}"
                )

    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "archive destination "
            f"(default: {DEFAULT_OUTPUT.relative_to(ROOT)})"
        ),
    )
    args = parser.parse_args()

    output = args.output.resolve()
    digest = build_archive(output)
    print(f"built {output}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
