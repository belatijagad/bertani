from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("bertani._rust", reason="the maturin extension has not been built")


def load_packager() -> ModuleType:
    path = ROOT / "scripts" / "package_rule_agent.py"
    spec = importlib.util.spec_from_file_location("package_rule_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_python_strategy_builds_a_self_contained_archive(tmp_path: Path) -> None:
    packager = load_packager()
    strategy = ROOT / "src" / "bertani_rules" / "strategies" / "simple.py"
    output = tmp_path / "simple.tar.gz"

    digest = packager.build_archive(output, strategy)

    assert len(digest) == 64
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        main = archive.extractfile("main.py")
        assert main is not None
        main_source = main.read().decode()
    assert {
        "main.py",
        "rules.py",
        "bertani_rules/strategy.py",
        "bertani_rules/strategies/current.py",
        "bertani_rules/strategies/simple.py",
    }.issubset(names)
    assert any(name.startswith("bertani/_rust") for name in names)
    assert "from bertani_rules.strategies.simple import build_policy" in main_source

    result = packager.smoke_test_archive(output)
    assert [state["status"] for state in result] == ["DONE", "DONE"]
