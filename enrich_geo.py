"""Enrich the pulled outlets with the STRUCTURAL peer factors that are actually
populated in the warehouse (audited fill rates): GPS (lat/long), pincode,
marketname, city. Single-catalog, bounded by the outlet-ids we already have.

These feed real peer axes the first pass lacked:
  catchment_density  <- from GPS (outlets per km^2)
  geo_unit           <- marketname / city
  affluence_band     <- pincode -> city-tier proxy (Trino cos; Colgate has no pincode)
"""
from __future__ import annotations

import polars as pl
from ch_client import run as chrun
from pull_trino import run as trun

TRINO = [11022, 11016, 10387, 10177]
COLGATE = 234032


def _df(cols, rows):
    return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None) if rows \
        else pl.DataFrame(schema=cols)


def trino_geo(cid: int, ids: list[int]) -> pl.DataFrame:
    idl = ",".join(str(i) for i in ids)
    cols, rows = trun(
        f"""SELECT id AS outletid, latitude AS lat, longitude AS lon, pincode,
                   marketname, city AS geocity, subcity
            FROM masterdb.dbo.f2klocations WHERE company={cid} AND id IN ({idl})""",
        timeout=120)
    return _df(cols, rows)


def colgate_geo(ids: list[int]) -> pl.DataFrame:
    idl = ",".join(str(i) for i in ids)
    m, rows = chrun(
        f"""SELECT ID AS outletid, toFloat64(Latitude) AS lat, toFloat64(Longitude) AS lon,
                   PinCode AS pincode, MarketName AS marketname, City AS geocity, SubCity AS subcity
            FROM unify.master_f2klocations WHERE Company={COLGATE} AND ID IN ({idl})""",
        timeout=180)
    return _df([c["name"] for c in m], rows).unique(subset=["outletid"], keep="first")


def main() -> None:
    base = pl.read_parquet("data/outlets_all.parquet").with_columns(pl.col("outletid").cast(pl.Int64))
    frames = []
    for cid in base.select(pl.col("company_id").unique()).to_series().to_list():
        ids = base.filter(pl.col("company_id") == cid).get_column("outletid").to_list()
        try:
            g = (colgate_geo(ids) if cid == COLGATE else trino_geo(cid, ids))
        except Exception as e:
            print(f"  !! company {cid}: {str(e)[:120]}", flush=True)
            continue
        if g.height:
            g = g.with_columns(pl.col("outletid").cast(pl.Int64),
                               pl.col("lat").cast(pl.Float64, strict=False),
                               pl.col("lon").cast(pl.Float64, strict=False),
                               pl.col("pincode").cast(pl.Utf8, strict=False))
            frames.append(g.with_columns(company_id=pl.lit(cid)))
        matched = g.height
        print(f"  company {cid}: geo matched {matched}/{len(ids)}", flush=True)

    geo = pl.concat(frames, how="diagonal_relaxed")
    # join on (company_id, outletid); keep first geo row per outlet
    geo = geo.unique(subset=["company_id", "outletid"], keep="first")
    out = base.join(geo, on=["company_id", "outletid"], how="left")
    out.write_parquet("data/outlets_geo.parquet")
    have = out.filter(pl.col("lat").is_not_null() & (pl.col("lat") != 0)).height
    print(f"\nWROTE data/outlets_geo.parquet — {out.height} rows, GPS on {have} "
          f"({round(100*have/out.height)}%)", flush=True)


if __name__ == "__main__":
    main()
