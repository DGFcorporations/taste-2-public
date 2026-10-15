import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """Fresh app on a throwaway LanceDB dataset per test."""
    db_dir = tmp_path / "lance_test"
    monkeypatch.setenv("TASTE2_DB", str(db_dir))
    monkeypatch.setenv("TASTE_EMBED", "hash")  # deterministic, offline-fast
    from taste.embed import get_embedder
    get_embedder.cache_clear()
    from taste.api import create_app
    return create_app(db_dir)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
