import uuid

import pytest
import sqlalchemy

from blunderless.analysis import cache
from blunderless.analysis.engine import LineEval, PositionAnalysis
from blunderless.db.session import make_engine, make_session_factory


def _db_reachable() -> bool:
    try:
        with make_engine().connect():
            return True
    except sqlalchemy.exc.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="postgres not reachable")


def _analysis(nodes: int, lines: int) -> PositionAnalysis:
    return PositionAnalysis(
        fen="",
        nodes=nodes,
        multipv=lines,
        lines=[
            LineEval(move=f"e2e{i + 3}", eval_cp=30 - 10 * i, mate_in=None, pv=[f"e2e{i + 3}"])
            for i in range(lines)
        ],
    )


@pytest.fixture
def db():
    with make_session_factory()() as session:
        yield session
        session.rollback()


def _key() -> str:
    return f"test_{uuid.uuid4().hex}"  # unique per test run, never collides with real data


def test_roundtrip(db):
    key = _key()
    cache.put(db, key, _analysis(40_000, 1))
    got = cache.get(db, key, 40_000, multipv=1)
    assert got is not None
    assert got.best.move == "e2e3"
    assert got.best.eval_cp == 30


def test_deep_request_never_gets_shallow_hit(db):
    key = _key()
    cache.put(db, key, _analysis(40_000, 1))
    assert cache.get(db, key, 350_000, multipv=1) is None  # different node count


def test_multipv_request_rejects_single_pv_entry(db):
    key = _key()
    cache.put(db, key, _analysis(40_000, 1))
    assert cache.get(db, key, 40_000, multipv=4) is None
    # but a MultiPV entry satisfies a single-PV request
    key2 = _key()
    cache.put(db, key2, _analysis(40_000, 4))
    assert cache.get(db, key2, 40_000, multipv=1) is not None


def test_multipv_upgrades_single_pv_entry(db):
    key = _key()
    cache.put(db, key, _analysis(40_000, 1))
    cache.put(db, key, _analysis(40_000, 4))  # strictly more information
    got = cache.get(db, key, 40_000, multipv=4)
    assert got is not None
    assert len(got.lines) == 4
    # and a weaker write never downgrades a richer entry
    cache.put(db, key, _analysis(40_000, 1))
    assert cache.get(db, key, 40_000, multipv=4) is not None


def test_get_many_batches(db):
    keys = [_key() for _ in range(3)]
    for k in keys[:2]:
        cache.put(db, k, _analysis(40_000, 1))
    found = cache.get_many(db, keys, 40_000)
    assert set(found) == set(keys[:2])
