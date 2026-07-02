"""Shared fixtures. Runs the engine once on the real pulled data."""
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import segment  # noqa: E402

DATA = next((p for p in (ROOT / "data" / "outlets_geo2.parquet", ROOT / "data" / "outlets_geo.parquet",
                         ROOT / "data" / "outlets_all.parquet",
                         ROOT / "data" / "outlets.parquet") if p.exists()),
            ROOT / "data" / "outlets.parquet")


@pytest.fixture(scope="session")
def result():
    if not DATA.exists():
        pytest.skip("no data parquet present — run pull_data.py / pull_colgate.py first")
    return segment.run_engine(str(DATA))


@pytest.fixture(scope="session")
def graded(result):
    return result.graded


def synth(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal outlets frame matching the pull schema, for edge tests."""
    cols = {
        "company_id": 1, "company_name": "TestCo", "outletid": 0, "regionname": "R",
        "territoryname": "T", "city": "C", "beatid": 1, "shoptypename": "Kirana Store",
        "channelname": "GT", "segmentationname": None, "bills": 0, "total_value": 0.0,
        "last_bill": None, "first_bill": None, "order_weeks": 0, "distinct_skus": 0,
        "line_value": 0.0, "has_data": False,
    }
    out = []
    for i, r in enumerate(rows):
        base = dict(cols)
        base["outletid"] = i + 1
        base.update(r)
        out.append(base)
    return pl.DataFrame(out)


def write_synth(tmp_path, rows) -> str:
    p = tmp_path / "synth.parquet"
    synth(rows).write_parquet(p)
    return str(p)
