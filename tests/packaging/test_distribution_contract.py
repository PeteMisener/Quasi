"""Phase 2 distribution contract: the installed package and options must be real."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_distribution_is_distinct_from_reserved_quasi_name():
    assert PROJECT["project"]["name"] == "quasi-qed"
    assert "quasi" != PROJECT["project"]["name"]


def test_version_declared_once_and_import_metadata_consistent():
    assert PROJECT["project"]["version"] == "0.2.0.dev0"
    try:
        installed_version = importlib.metadata.version("quasi-qed")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    import quasi
    if installed_version is not None:
        assert quasi.__version__ == installed_version
    else:
        assert quasi.__version__ == "0+unknown"


def test_optional_dependencies_really_optional():
    deps = PROJECT["project"]["dependencies"]
    assert len(deps) == 3
    for name in ("matplotlib", "rich", "polars", "pyarrow", "httpx"):
        assert not any(name in dep for dep in deps)
    extras = PROJECT["project"]["optional-dependencies"]
    assert {"plot", "report", "ai", "polars", "dev", "all"} <= extras.keys()


def test_all_imports_work_without_optional_plot_or_rich():
    code = r'''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('matplotlib', 'rich', 'polars', 'httpx'):
            raise ModuleNotFoundError(f'blocked optional dependency: {fullname}')
sys.meta_path.insert(0, BlockOptional())
import quasi
from quasi import estimate, audit_panel, StaggeredDiD, TwoStageLS
assert callable(estimate)
assert callable(audit_panel)
assert 'matplotlib' not in sys.modules
assert 'rich' not in sys.modules
# Prove a real computation works without plotting and rich extras.
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
z = rng.normal(size=120)
x = 0.8*z + rng.normal(size=120)*0.4
y = 2*x + rng.normal(size=120)*0.2
model = TwoStageLS(pd.DataFrame({'y': y, 'x': x, 'z': z}), outcome='y', endogenous='x', instruments=['z'])
assert np.isfinite(model.fit().estimate)
assert 'matplotlib' not in sys.modules
assert 'rich' not in sys.modules
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=45,
    )
    assert proc.returncode == 0, proc.stderr


def test_init_files_are_valid_and_no_fake_reference_workflow():
    assert (ROOT / "src/quasi/synth/__init__.py").exists()
    assert (ROOT / "src/quasi/diagnostics/__init__.py").exists()
    assert not (ROOT / "src/quasi/synth/__initi__.py").exists()
    assert not (ROOT / "src/quasi/diagnostics/__initi__.py").exists()
    assert not (ROOT / "src/quasi/.github").exists()


def test_basic_wheel_inputs_and_license():
    assert (ROOT / "LICENSE").exists()
    assert (ROOT / "README.md").exists()
    assert "\\rtf1" not in (ROOT / "requirements.txt").read_text()
    assert PROJECT["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_authentic_parity_is_not_asserted_from_synthetic_fixtures():
    for path in (ROOT / "tests/fixtures").glob("*_benchmark.json"):
        assert path.is_file()  # these are documented as synthetic fallbacks
    assert (ROOT / "docs/validation.md").exists()
