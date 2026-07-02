"""Pull real Colgate (Colgate Palmolive India, company 234032) data from the
colpal ClickHouse tenant, and INCLUDE RETURNS.

Colgate has no FA SecondaryInvoice — its secondary sales live in
`NonFAInvoiceDetail` (distributor-uploaded invoices), 30M rows. InvoiceType
decodes as: 0 = Sales, 1 = Credit note ('C'), 2 = Return ('R'). We fold 1+2 as
returns (value flowing back), and keep the same per-outlet feature schema as the
Trino pull plus `returns_value` / `return_bills`.

Everything is one single-table aggregation over NonFAInvoiceDetail (ClickHouse
does 3-region aggregates in ~0.5s), then master joins for typing.
Output schema matches data/outlets.parquet + returns columns.
"""
from __future__ import annotations

import polars as pl

from ch_client import run

CID = 234032
NAME = "Colgate (India)"
WINDOW = "2026-04-01"        # 3-month window, matches the Trino pull
TARGET_PER_REGION = 700
NODATA = 300
RETURN_TYPES = "(1, 2)"     # 1 = credit note (C), 2 = return (R)


def df(cols, rows) -> pl.DataFrame:
    names = [c["name"] for c in cols]
    return pl.DataFrame(rows, schema=names, orient="row", infer_schema_length=None) if rows \
        else pl.DataFrame(schema=names)


def pick_regions() -> list[int]:
    _, rows = run(
        f"""SELECT RegionId, uniqExact(LocationId) n
            FROM unify.NonFAInvoiceDetail
            WHERE CompanyId={CID} AND InvoiceType=0 AND InvoiceDate>='{WINDOW}'
              AND RegionId>0
            GROUP BY RegionId ORDER BY n DESC""")
    regs = [(int(r[0]), int(r[1])) for r in rows]
    # every region is huge; pick 3 spread across the size distribution (25/50/75%)
    idx = [len(regs) // 4, len(regs) // 2, (3 * len(regs)) // 4]
    return [regs[i][0] for i in idx]


def region_names(ids: list[int]) -> dict[int, str]:
    idl = ",".join(str(i) for i in ids)
    _, rows = run(f"SELECT Id, Name FROM unify.master_regions WHERE Id IN ({idl})")
    return {int(r[0]): r[1] for r in rows}


def channel_map() -> dict[int, str]:
    m, rows = run(f"SELECT * FROM unify.master_channels WHERE CompanyId={CID}")
    names = [c["name"] for c in m]
    ci = names.index("ChannelId") if "ChannelId" in names else 2
    ni = names.index("Name") if "Name" in names else 3
    return {int(r[ci]): r[ni] for r in rows}


def spine(region_ids: list[int]) -> pl.DataFrame:
    idl = ",".join(str(i) for i in region_ids)
    m, rows = run(
        f"""SELECT LocationId AS outletid, RegionId AS region_id,
                   uniqExactIf(InvoiceNumber, InvoiceType=0) AS bills,
                   round(sumIf(OrderInRevenue, InvoiceType=0), 2) AS total_value,
                   toString(maxIf(InvoiceDate, InvoiceType=0)) AS last_bill,
                   toString(minIf(InvoiceDate, InvoiceType=0)) AS first_bill,
                   uniqExactIf(toMonday(InvoiceDate), InvoiceType=0) AS order_weeks,
                   uniqExactIf(ProductId, InvoiceType=0) AS distinct_skus,
                   round(sumIf(OrderInRevenue, InvoiceType=0), 2) AS line_value,
                   round(sumIf(OrderInRevenue, InvoiceType IN {RETURN_TYPES}), 2) AS returns_value,
                   uniqExactIf(InvoiceNumber, InvoiceType IN {RETURN_TYPES}) AS return_bills
            FROM unify.NonFAInvoiceDetail
            WHERE CompanyId={CID} AND InvoiceDate>='{WINDOW}' AND RegionId IN ({idl})
            GROUP BY LocationId, RegionId
            HAVING bills > 0
            ORDER BY bills DESC
            LIMIT {TARGET_PER_REGION} BY RegionId""", timeout=300)
    return df(m, rows)


def master_for(ids: list[int]) -> pl.DataFrame:
    if not ids:
        return pl.DataFrame()
    idl = ",".join(str(i) for i in ids)
    m, rows = run(
        f"""SELECT ID AS outletid, ShopType AS shoptypename, OutletChannel AS channel_code,
                   City AS city, BeatId AS beatid, Segmentation AS seg
            FROM unify.master_f2klocations
            WHERE Company={CID} AND ID IN ({idl})""", timeout=180)
    # master has ~1.6 rows/outlet — keep one row per outlet so joins don't fan out
    return df(m, rows).unique(subset=["outletid"], keep="first")


def nodata(selling_ids: list[int], beats: list[int]) -> pl.DataFrame:
    if not beats:
        return pl.DataFrame()
    bl = ",".join(str(b) for b in beats if b)
    excl = ",".join(str(i) for i in selling_ids)
    m, rows = run(
        f"""SELECT ID AS outletid, ShopType AS shoptypename, OutletChannel AS channel_code,
                   City AS city, BeatId AS beatid, Segmentation AS seg
            FROM unify.master_f2klocations
            WHERE Company={CID} AND BeatId IN ({bl}) AND ID NOT IN ({excl})
            LIMIT {NODATA * 3}""", timeout=180)
    return df(m, rows).unique(subset=["outletid"], keep="first").head(NODATA)


def main() -> None:
    region_ids = pick_regions()
    rnames = region_names(region_ids)
    chan = channel_map()
    print(f"[Colgate] regions: {[rnames.get(i, i) for i in region_ids]}", flush=True)

    sp = spine(region_ids)
    print(f"  spine rows (capped {TARGET_PER_REGION}/region): {sp.height}", flush=True)
    # ClickHouse returns uniqExact()/counts as strings -> cast to real numerics
    sp = sp.with_columns(
        pl.col("outletid").cast(pl.Int64),
        pl.col("bills").cast(pl.Int64, strict=False),
        pl.col("order_weeks").cast(pl.Int64, strict=False),
        pl.col("distinct_skus").cast(pl.Int64, strict=False),
        pl.col("return_bills").cast(pl.Int64, strict=False),
        pl.col("total_value").cast(pl.Float64, strict=False),
        pl.col("line_value").cast(pl.Float64, strict=False),
        pl.col("returns_value").cast(pl.Float64, strict=False),
    )
    selling_ids = sp.get_column("outletid").to_list()
    mst = master_for(selling_ids).with_columns(pl.col("outletid").cast(pl.Int64))

    sell = sp.join(mst, on="outletid", how="left").with_columns(
        regionname=pl.col("region_id").cast(pl.Int64).replace_strict(rnames, default="Region?"),
        channelname=pl.col("channel_code").cast(pl.Int64, strict=False).replace_strict(chan, default=None),
        has_data=pl.lit(True),
    )
    # beat -> region (mode) so no-data outlets in the same beats inherit a region
    beat_region = dict(sell.group_by("beatid").agg(pl.col("regionname").mode().first())
                       .iter_rows())
    beats = [b for b in sell.get_column("beatid").unique().to_list() if b is not None]

    nod = nodata(selling_ids, beats)
    if nod.height:
        nod = nod.with_columns(pl.col("outletid").cast(pl.Int64)).with_columns(
            regionname=pl.col("beatid").replace_strict(beat_region, default="Region?"),
            channelname=pl.col("channel_code").cast(pl.Int64, strict=False).replace_strict(chan, default=None),
            bills=pl.lit(0), total_value=pl.lit(0.0), last_bill=pl.lit(None), first_bill=pl.lit(None),
            order_weeks=pl.lit(0), distinct_skus=pl.lit(0), line_value=pl.lit(0.0),
            returns_value=pl.lit(0.0), return_bills=pl.lit(0), has_data=pl.lit(False),
        )
    print(f"  selling={sell.height} nodata={nod.height}", flush=True)

    out = pl.concat([sell, nod], how="diagonal_relaxed").with_columns(
        company_id=pl.lit(CID), company_name=pl.lit(NAME),
        territoryname=pl.lit(None, dtype=pl.Utf8),
        segmentationname=pl.col("seg").cast(pl.Utf8),
    )
    cols = ["company_id", "company_name", "outletid", "regionname", "territoryname", "city",
            "beatid", "shoptypename", "channelname", "segmentationname",
            "bills", "total_value", "last_bill", "first_bill", "order_weeks",
            "distinct_skus", "line_value", "has_data", "returns_value", "return_bills"]
    out = out.select([c for c in cols if c in out.columns])
    out.write_parquet("data/colgate.parquet")
    print(f"WROTE data/colgate.parquet — {out.height} outlets "
          f"({int(sell.height)} selling, {int(nod.height)} cold-start)", flush=True)

    # combine with the Trino 4-company pull (which has no returns cols -> null)
    base = pl.read_parquet("data/outlets.parquet")
    combined = pl.concat([base, out], how="diagonal_relaxed")
    combined.write_parquet("data/outlets_all.parquet")
    print(f"WROTE data/outlets_all.parquet — {combined.height} outlets, "
          f"{combined.select(pl.col('company_name').n_unique()).item()} companies", flush=True)


if __name__ == "__main__":
    main()
