import pytest


@pytest.fixture(autouse=True)
def _one_chain_after_the_other(monkeypatch):
    """Stem chains are made in this process unless a test asks for workers: what a test puts in the
    place of a function (monkeypatch) is not there in a process started afresh."""
    monkeypatch.setenv("MOOZ_WORKERS", "1")
