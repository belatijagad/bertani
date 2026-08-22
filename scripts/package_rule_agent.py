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
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "bertani"
RULE_SOURCE = ROOT / "src" / "bertani_rules" / "agent.py"
RULE_PACKAGE_SOURCE = ROOT / "src" / "bertani_rules"
DEFAULT_OUTPUT = ROOT / "dist" / "rule_based_submission.tar.gz"

MODULES = (
    "vec_env.py",
    "opening.py",
    "market.py",
    "tasks.py",
    "rule_based.py",
    "kaggle_agent.py",
)


def _main_module(rule_source: Path) -> str:
    """Return a package import when the strategy lives in ``src``."""

    try:
        relative = rule_source.resolve().relative_to(RULE_PACKAGE_SOURCE.resolve())
    except ValueError:
        return "rules"
    return ".".join(("bertani_rules", *relative.with_suffix("").parts))


def _main_payload(rule_source: Path) -> bytes:
    module = _main_module(rule_source)
    return f'''"""Bertani rule-based Kaggriculture submission."""
from bertani.kaggle_agent import make_agent
from {module} import build_policy

agent = make_agent(build_policy)
__all__ = ["agent"]
'''.encode()


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
        path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise RuntimeError(
            f"bertani._rust did not resolve to a CPython extension: {path}"
        )
    return path


def archive_members(rule_source: Path = RULE_SOURCE) -> dict[str, bytes]:
    """Collect the Python agent and its native route solver."""

    if not rule_source.is_file():
        raise FileNotFoundError(f"rule strategy is missing: {rule_source}")

    rule_payload = rule_source.read_bytes()
    compile(rule_payload, str(rule_source), "exec")

    native = _native_extension()
    native_member = f"bertani/{native.name}"

    members: dict[str, bytes] = {
        "main.py": _main_payload(rule_source),
        "bertani/__init__.py": (SOURCE / "__init__.py").read_bytes(),
        "rules.py": rule_payload,
        native_member: native.read_bytes(),
    }

    for name in MODULES:
        path = SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(f"required rule-agent module is missing: {path}")
        payload = path.read_bytes()
        compile(payload, str(path), "exec")
        members[f"bertani/{name}"] = payload

    for name in ("__init__.py", "agent.py", "strategy.py"):
        path = RULE_PACKAGE_SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(f"required rule-agent module is missing: {path}")
        payload = path.read_bytes()
        compile(payload, str(path), "exec")
        members[f"bertani_rules/{name}"] = payload

    strategy_directory = RULE_PACKAGE_SOURCE / "strategies"
    for path in sorted(strategy_directory.glob("*.py")):
        payload = path.read_bytes()
        compile(payload, str(path), "exec")
        members[f"bertani_rules/strategies/{path.name}"] = payload

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


def build_archive(output: Path, rule_source: Path = RULE_SOURCE) -> str:
    """Write an order-stable, timestamp-free tar.gz and return its SHA-256."""

    members = archive_members(rule_source)
    output.parent.mkdir(parents=True, exist_ok=True)

    with (
        output.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            mtime=0,
        ) as zipped,
        tarfile.open(
            fileobj=zipped,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive,
    ):
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
                raise RuntimeError(f"unexpected non-file archive member: {member.name}")

    return hashlib.sha256(payload).hexdigest()


def smoke_test_archive(output: Path) -> list[dict[str, Any]]:
    """Run the packaged ``main.py`` in the original Python environment.

    Extraction happens in a fresh directory and a fresh interpreter so local
    editable source modules cannot hide missing archive members.
    """

    program = """
import json
from pathlib import Path

from kaggle_environments import make

import bertani
import rules

root = Path.cwd().resolve()
assert Path(bertani.__file__).resolve().is_relative_to(root)
assert Path(rules.__file__).resolve().is_relative_to(root)

environment = make(
    "kaggriculture",
    configuration={"episodeSteps": 720, "seed": 11},
    debug=True,
)
environment.run(["main.py", "pass"])
result = [
    {"status": str(state.status), "reward": float(state.reward or 0.0)}
    for state in environment.steps[-1]
]
print(json.dumps(result))
"""
    with tempfile.TemporaryDirectory(prefix="bertani-submission-") as directory:
        extracted = Path(directory)
        with tarfile.open(output, "r:gz") as archive:
            archive.extractall(extracted, filter="data")
        completed = subprocess.run(
            (sys.executable, "-c", program),
            cwd=extracted,
            check=True,
            capture_output=True,
            text=True,
        )

    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if not isinstance(result, list) or any(
        state.get("status") != "DONE" for state in result
    ):
        raise RuntimeError(f"submission smoke test did not finish: {result!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        type=Path,
        default=RULE_SOURCE,
        help="Python module defining build_policy (default: current strategy)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(f"archive destination (default: {DEFAULT_OUTPUT.relative_to(ROOT)})"),
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="skip running the isolated archive in kaggle-environments",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    strategy = args.strategy.resolve()
    digest = build_archive(output, strategy)
    print(f"built {output}")
    print(f"sha256 {digest}")
    if not args.no_smoke_test:
        result = smoke_test_archive(output)
        print(
            "smoke test "
            + ", ".join(
                f"seat {seat}: {state['status']} reward={state['reward']:.0f}"
                for seat, state in enumerate(result)
            )
        )


if __name__ == "__main__":
    main()
