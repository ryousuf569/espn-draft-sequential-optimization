# Each model package imports its siblings flatly (`from board import Board`), so a
# package directory has to be on sys.path before its modules resolve. Both packages
# also ship a module named config, so whichever imports second would otherwise bind
# the other's. load_package() caches each package's modules under a private prefix
# and restores sys.modules afterwards, which lets one pytest process import both.

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "nba.sqlite"

# the season every fixture below is built against
SEASON = "2025-26"


# The DB is a release asset, never committed, so CI runs without one. Tests that
# need it skip rather than fail, and the pure-logic tests carry the suite.
def pytest_configure(config):
    config.addinivalue_line("markers", "needs_db: requires data/nba.sqlite")


def pytest_collection_modifyitems(config, items):
    if DB_PATH.exists():
        return

    skip = pytest.mark.skip(reason=f"no database at {DB_PATH}")
    for item in items:
        if "needs_db" in item.keywords:
            item.add_marker(skip)


# Import a model package's modules under a private prefix, so two packages that both
# define `config` can coexist. Mirrors what models/decision/config.py already does
# for models/behaviour, applied to whichever package a test file asks for.
def load_package(package, names):
    directory = str(ROOT / "models" / package)
    prefix = f"_{package}_"

    saved = {k: sys.modules.get(k) for k in ("config", *names)}
    for key, module in list(sys.modules.items()):
        if key.startswith(prefix):
            sys.modules[key[len(prefix):]] = module

    sys.path.insert(0, directory)
    try:
        loaded = {}
        for name in ("config", *names):
            loaded[name] = importlib.import_module(name)
        for name, module in loaded.items():
            sys.modules[prefix + name] = module
        return loaded
    finally:
        sys.path.remove(directory)
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


@pytest.fixture(scope="session")
def conn():
    if not DB_PATH.exists():
        pytest.skip("no database")
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()
