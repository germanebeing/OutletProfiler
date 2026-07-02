"""Pull a real, bounded multi-company dataset from the FieldAssist warehouse
for the segmentation-lab validation. Read-only aggregation queries.

Hard-won lesson: the 12GB blow-up + multi-minute hangs were caused by
CROSS-CATALOG joins (transactiondb line-items JOIN famasters entity) — Trino
can't push those down, so it streamed the whole line-item table into memory.

Design: every warehouse query is SINGLE-CATALOG (pure transactiondb OR pure
famasters) so SQL Server executes it natively with indexes; all joins happen in
Polars. Whole-company item aggregate then runs in ~2s instead of hanging.
"""
from __future__ import annotations

import json

import polars as pl
from pull_trino import run

WINDOW = "DATE '2026-04-01'"  # last ~3 months (>= 1 reorder cycle)
COMPANIES = {
    11022: "CG Corp (Wai Wai)",
    11016: "Anchor",
    10387: "Hamdard",
    10177: "Everest Spices",
}
TARGET_PER_REGION = 700    # cap selling outlets per region
NODATA_PER_COMPANY = 250   # cold-start sample


def df(cols, rows) -> pl.DataFrame:
    # infer_schema_length=None -> scan all rows, so text cols that start null
    # (e.g. city/territory) aren't mis-typed and then rejected on a later string.
    if not rows:
        return pl.DataFrame(schema=cols)
    return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None)


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def pick_regions(cid: int) -> list[str]:
    cols, rows = run(
        f"""SELECT b.regionname, count(distinct s.outletid) n
            FROM transactiondb.dbo.secondarysales s
            JOIN famasters.dbo.buyersellerentity b ON s.outletid=b.entityid AND b.companyid={cid}
            WHERE s.companyid={cid} AND s.createdat>={WINDOW}
              AND b.regionname IS NOT NULL AND b.regionname<>''
            GROUP BY b.regionname ORDER BY n DESC""",
        timeout=180,
    )
    regs = [(r[0], int(r[1])) for r in rows]
    elig = [r for r in regs if r[1] >= 150]
    band = [r for r in elig if r[1] <= 1500]            # goldilocks
    chosen = list(band[:3])
    if len(chosen) < 3:                                 # top up to >=3 (client wants >=3)
        extra = sorted([r for r in elig if r not in band], key=lambda x: x[1])
        chosen = (chosen + extra)[:3]
    if len(chosen) < 3:                                 # tiny company: take whatever exists
        chosen = (chosen + [r for r in regs if r not in chosen])[:3]
    return [c[0] for c in chosen]


def master(cid: int, regions: list[str]) -> pl.DataFrame:
    """All registered outlets in the chosen regions — pure famasters (fast)."""
    rlist = ", ".join(_q(r) for r in regions)
    cols, rows = run(
        f"""SELECT b.entityid AS outletid, b.regionname, b.territoryname, b.city, b.beatid,
                   b.shoptypename, b.channelname, b.segmentationname
            FROM famasters.dbo.buyersellerentity b
            WHERE b.companyid={cid} AND b.regionname IN ({rlist})""",
        timeout=180,
    )
    return df(cols, rows)


def spine(cid: int) -> pl.DataFrame:
    """Sales spine per outlet, whole company — pure transactiondb (pushdown)."""
    cols, rows = run(
        f"""SELECT outletid, count(distinct id) bills, sum(invoiceamount) total_value,
                   max(createdat) last_bill, min(createdat) first_bill,
                   count(distinct cast(date_trunc('week', createdat) as date)) order_weeks
            FROM transactiondb.dbo.secondarysales
            WHERE companyid={cid} AND createdat>={WINDOW}
            GROUP BY outletid""",
        timeout=240,
    )
    return df(cols, rows)


def srange(cid: int) -> pl.DataFrame:
    """SKU breadth + line value per outlet, whole company — pure transactiondb."""
    cols, rows = run(
        f"""SELECT s.outletid,
                   approx_distinct(i.productid) distinct_skus,
                   sum(i.dispatchedquantity * coalesce(try_cast(i.billedptr AS double), 0)) line_value
            FROM transactiondb.dbo.secondarysales s
            JOIN transactiondb.dbo.secondarysaleitems i ON i.secondarysaleid=s.id
            WHERE s.companyid={cid} AND s.createdat>={WINDOW}
            GROUP BY s.outletid""",
        timeout=300,
    )
    return df(cols, rows)


def pull_company(cid: int, name: str) -> tuple[pl.DataFrame, dict]:
    regions = pick_regions(cid)
    print(f"[{name}] regions: {regions}", flush=True)
    mst = master(cid, regions)
    sp = spine(cid).with_columns(has_data=pl.lit(True))
    rg = srange(cid)
    print(f"  master={mst.height} spine={sp.height} range={rg.height}", flush=True)

    # join everything in Polars; master is region-scoped so it bounds the result
    feat = sp.join(rg, on="outletid", how="left")
    d = mst.join(feat, on="outletid", how="left")
    d = d.with_columns(
        has_data=pl.col("has_data").fill_null(False),
        bills=pl.col("bills").fill_null(0),
        total_value=pl.col("total_value").fill_null(0.0),
        distinct_skus=pl.col("distinct_skus").fill_null(0),
        line_value=pl.col("line_value").fill_null(0.0),
        order_weeks=pl.col("order_weeks").fill_null(0),
    )

    selling = d.filter(pl.col("has_data"))
    # cap selling per region by bills
    selling = (
        selling.with_columns(
            pl.col("bills").rank("ordinal", descending=True).over("regionname").alias("_rk"))
        .filter(pl.col("_rk") <= TARGET_PER_REGION).drop("_rk")
    )
    nodata = d.filter(~pl.col("has_data"))
    if nodata.height > NODATA_PER_COMPANY:
        nodata = nodata.sample(NODATA_PER_COMPANY, seed=7)

    out = pl.concat([selling, nodata], how="diagonal_relaxed").with_columns(
        company_id=pl.lit(cid), company_name=pl.lit(name))
    info = {"company_id": cid, "regions": regions,
            "selling_outlets": int(selling.height), "nodata_outlets": int(nodata.height)}
    print(f"  => selling={selling.height} nodata={nodata.height}", flush=True)
    return out, info


def main() -> None:
    frames, summary = [], {}
    for cid, name in COMPANIES.items():
        try:
            out, info = pull_company(cid, name)
        except Exception as e:
            print(f"  !! {name}: {str(e)[:150]}", flush=True)
            continue
        frames.append(out)
        summary[name] = info

    allcols = ["company_id", "company_name", "outletid", "regionname", "territoryname", "city",
               "beatid", "shoptypename", "channelname", "segmentationname",
               "bills", "total_value", "last_bill", "first_bill", "order_weeks",
               "distinct_skus", "line_value", "has_data"]
    full = pl.concat([f.select([c for c in allcols if c in f.columns]) for f in frames],
                     how="diagonal_relaxed")
    full.write_parquet("data/outlets.parquet")
    with open("data/companies.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nWROTE data/outlets.parquet — {full.height} outlets total", flush=True)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
