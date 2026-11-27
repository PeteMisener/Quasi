"""Pytest configuration, shared test fixtures, and tolerance thresholds."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd
import pytest

from tests.fixtures.seed_data import (
    ensure_benchmark_files_exist,
    get_mpdta_benchmark,
    get_prop99_benchmark,
    get_senate_benchmark,
)


@pytest.fixture(scope="session", autouse=True)
def initialize_fixtures() -> None:
    """Generate golden JSON fixtures on disk before tests run."""
    ensure_benchmark_files_exist()


@pytest.fixture(scope="session")
def mpdta_fixture() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return get_mpdta_benchmark()


@pytest.fixture(scope="session")
def senate_fixture() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return get_senate_benchmark()


@pytest.fixture(scope="session")
def prop99_fixture() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return get_prop99_benchmark()