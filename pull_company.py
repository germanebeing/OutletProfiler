"""Onboard a NEW company from any FieldAssist Trino host, on demand.

Real clients diverge in their data model. Two are handled:
  - "bse": secondarysales.outletid = famasters.buyersellerentity.entityid
           (region from buyersellerentity.regionname) — the 4 base companies.
  - "f2k": secondarysales.outletid = masterdb.f2klocations.id
           (geo unit = f2klocations.state) — e.g. GIL Live.
We detect which by test-joining, then pull master + region accordingly. The
sales spine / SKU range are keyed on outletid alone, so they are model-agnostic.
Returns a frame matching data/outlets_geo2.parquet.
"""
from __future__ import annotations

import json
import time

import httpx
import polars as pl

H = {"X-Trino-User": "admin", "X-Presto-User": "admin"}
WINDOW = "DATE '2026-04-01'"
TARGET_PER_REGION = 700
CHANNEL_ENUM = {0: "Others", 1: "GT", 2: "MT", 3: "HoReCa", 4: "Semi MT", 5: "Institutional",
                6: "CSD", 7: "Key Accounts", 8: "Direct Dealers", 10: "Work Place", 12: "QSR", 13: "Pharmacy"}


def run_host(host: str, sql: str, timeout: int = 240):
    base = f"http://{host}:8080"
    r = httpx.post(base + "/v1/statement", content=sql.encode(), headers=H, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    cols, rows, guard = [], [], 0
    while True:
        guard += 1
        if guard > 5000:
            break
        if j.get("columns") and not cols:
            cols = [c["name"] for c in j["columns"]]
        if j.get("data"):
            rows.extend(j["data"])
        if j.get("error"):
            raise RuntimeError(json.dumps(j["error"].get("message", j["error"]))[:200])
        nu = j.get("nextUri")
        if not nu:
            break
        time.sleep(0.06)
        rr = httpx.get(nu, headers=H, timeout=timeout)
        rr.raise_for_status()
        j = rr.json()
    return cols, rows


def _df(cols, rows):
    return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None) if rows \
        else pl.DataFrame(schema=cols)


def _q(s):
    return "'" + s.replace("'", "''") + "'"


def detect_model(host, cid):
    _, r = run_host(host, f"""SELECT count(*) FROM transactiondb.dbo.secondarysales s
        JOIN famasters.dbo.buyersellerentity b ON s.outletid=b.entityid AND b.companyid={cid}
        WHERE s.companyid={cid} AND s.createdat>={WINDOW}""", 120)
    return "bse" if (r and int(r[0][0]) > 0) else "f2k"


def _top3(regs):
    regs = [(x[0], int(x[1])) for x in regs if x[0]]
    elig = [x for x in regs if x[1] >= 150]
    band = [x for x in elig if x[1] <= 1500]
    chosen = list(band[:3])
    if len(chosen) < 3:
        chosen = (chosen + sorted([x for x in elig if x not in band], key=lambda z: z[1]))[:3]
    if len(chosen) < 3:
        chosen = (chosen + [x for x in regs if x not in chosen])[:3]
    return [c[0] for c in chosen]


def _region_counts(host, cid, model):
    """All regions for the company, ranked by distinct selling outlets."""
    if model == "bse":
        _, r = run_host(host, f"""SELECT b.regionname, count(distinct s.outletid) n
            FROM transactiondb.dbo.secondarysales s
            JOIN famasters.dbo.buyersellerentity b ON s.outletid=b.entityid AND b.companyid={cid}
            WHERE s.companyid={cid} AND s.createdat>={WINDOW} AND b.regionname IS NOT NULL AND b.regionname<>''
            GROUP BY b.regionname ORDER BY n DESC""", 180)
    else:
        _, r = run_host(host, f"""SELECT f.state, count(distinct s.outletid) n
            FROM transactiondb.dbo.secondarysales s
            JOIN masterdb.dbo.f2klocations f ON f.id=s.outletid AND f.company={cid}
            WHERE s.companyid={cid} AND s.createdat>={WINDOW} AND f.state IS NOT NULL AND f.state<>''
            GROUP BY f.state ORDER BY n DESC""", 200)
    return r


def pick_regions(host, cid, model=None):
    model = model or detect_model(host, cid)
    return _top3(_region_counts(host, cid, model))


def _outlet_images(host, cid, outletids):
    """Storefront photo id per outlet from the outlet master (masterdb.f2klocations,
    keyed on company+id — covers every data model). The imageid is turned into a
    downloadable URL at type time via PROFILER_IMAGE_URL_TEMPLATE. NOTE: shelf /
    assortment images (transactiondb.flattenedfaimagedetects) are intentionally NOT
    used here — those are reserved for the future assortment-segmentation layer."""
    empty = pl.DataFrame(schema={"outletid": pl.Int64, "image_id": pl.Utf8})
    if not len(outletids):
        return empty
    try:
        c, r = run_host(host, f"""SELECT id AS outletid, imageid AS image_id
            FROM masterdb.dbo.f2klocations
            WHERE company={cid} AND imageid IS NOT NULL AND imageid<>''""", 300)
    except Exception:
        return empty
    d = _df(c, r)
    if d.height == 0:
        return empty
    keep = set(int(x) for x in outletids)
    return (d.with_columns(pl.col("outletid").cast(pl.Int64))
             .filter(pl.col("outletid").is_in(list(keep)))
             .unique(subset="outletid").select("outletid", "image_id"))


def list_all_regions(host, cid, model=None):
    """Every region the company sells in, with selling-outlet counts — for the
    operator to choose from at onboarding (or 'all')."""
    model = model or detect_model(host, cid)
    return [{"name": x[0], "n": int(x[1])} for x in _region_counts(host, cid, model) if x[0]]


def _master(host, cid, regions, model):
    rlist = ", ".join(_q(r) for r in regions)
    if model == "bse":
        mc, mr = run_host(host, f"""SELECT b.entityid AS outletid, b.regionname, b.territoryname, b.city,
            b.beatid, b.shoptypename, b.channelname, b.segmentationname
            FROM famasters.dbo.buyersellerentity b WHERE b.companyid={cid} AND b.regionname IN ({rlist})""", 180)
        return _df(mc, mr)
    mc, mr = run_host(host, f"""SELECT f.id AS outletid, f.state AS regionname, f.city,
        f.beatid, f.shoptype AS shoptypename, f.outletchannel AS channel_code,
        cast(f.segmentation as varchar) AS segmentationname
        FROM masterdb.dbo.f2klocations f WHERE f.company={cid} AND f.state IN ({rlist})""", 200)
    d = _df(mc, mr)
    if d.height:
        d = d.with_columns(
            channelname=pl.col("channel_code").cast(pl.Int64, strict=False)
            .replace_strict(CHANNEL_ENUM, default="GT"),
            territoryname=pl.lit(None, dtype=pl.Utf8)).drop("channel_code")
    return d


def pull_one(host: str, cid: int, name: str, regions: list[str] | None = None,
             with_images: bool = False):
    model = detect_model(host, cid)
    regions = [r for r in (regions or []) if r] or pick_regions(host, cid, model)
    if not regions:
        raise RuntimeError("no regions with sales found for this company id")
    mst = _master(host, cid, regions, model)
    sc, sr = run_host(host, f"""SELECT outletid, count(distinct id) bills, sum(invoiceamount) total_value,
        max(createdat) last_bill, min(createdat) first_bill,
        count(distinct cast(date_trunc('week', createdat) as date)) order_weeks
        FROM transactiondb.dbo.secondarysales WHERE companyid={cid} AND createdat>={WINDOW} GROUP BY outletid""", 300)
    sp = _df(sc, sr).with_columns(has_data=pl.lit(True))
    rc, rr = run_host(host, f"""SELECT s.outletid, approx_distinct(i.productid) distinct_skus,
        sum(i.dispatchedquantity * coalesce(try_cast(i.billedptr AS double), 0)) line_value
        FROM transactiondb.dbo.secondarysales s
        JOIN transactiondb.dbo.secondarysaleitems i ON i.secondarysaleid=s.id
        WHERE s.companyid={cid} AND s.createdat>={WINDOW} GROUP BY s.outletid""", 360)
    rg = _df(rc, rr)
    if sp.height == 0 or mst.height == 0:
        raise RuntimeError("no sales/master rows returned (model=%s)" % model)

    feat = sp.join(rg, on="outletid", how="left")
    d = mst.with_columns(pl.col("outletid").cast(pl.Int64)).join(
        feat.with_columns(pl.col("outletid").cast(pl.Int64)), on="outletid", how="left")
    d = d.with_columns(
        has_data=pl.col("has_data").fill_null(False),
        bills=pl.col("bills").cast(pl.Float64, strict=False).fill_null(0),
        total_value=pl.col("total_value").cast(pl.Float64, strict=False).fill_null(0.0),
        distinct_skus=pl.col("distinct_skus").cast(pl.Float64, strict=False).fill_null(0),
        line_value=pl.col("line_value").cast(pl.Float64, strict=False).fill_null(0.0),
        order_weeks=pl.col("order_weeks").cast(pl.Float64, strict=False).fill_null(0))
    selling = d.filter(pl.col("has_data")).with_columns(
        pl.col("bills").rank("ordinal", descending=True).over("regionname").alias("_rk")
    ).filter(pl.col("_rk") <= TARGET_PER_REGION).drop("_rk")
    nod = d.filter(~pl.col("has_data"))
    if nod.height > 300:
        nod = nod.sample(300, seed=7)
    out = pl.concat([selling, nod], how="diagonal_relaxed").with_columns(
        company_id=pl.lit(int(cid)), company_name=pl.lit(name),
        returns_value=pl.lit(0.0), return_bills=pl.lit(0),
        lat=pl.lit(None, dtype=pl.Float64), lon=pl.lit(None, dtype=pl.Float64),
        pincode=pl.lit(None, dtype=pl.Utf8), marketname=pl.lit(None, dtype=pl.Utf8),
        geocity=pl.lit(None, dtype=pl.Utf8), subcity=pl.lit(None, dtype=pl.Utf8),
        affluence_tier=pl.lit("na"), district=pl.lit(None, dtype=pl.Utf8))
    if with_images:
        ids = out.get_column("outletid").cast(pl.Int64).to_list()
        iu = _outlet_images(host, cid, ids)
        out = out.with_columns(pl.col("outletid").cast(pl.Int64)).join(iu, on="outletid", how="left")
    keep = ["company_id", "company_name", "outletid", "regionname", "territoryname", "city", "beatid",
            "shoptypename", "channelname", "segmentationname", "bills", "total_value", "last_bill",
            "first_bill", "order_weeks", "distinct_skus", "line_value", "has_data", "returns_value",
            "return_bills", "lat", "lon", "pincode", "marketname", "geocity", "subcity",
            "affluence_tier", "district", "image_id"]
    for c in keep:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).alias(c))
    return out.select(keep), regions, int(selling.height)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--regions":
        # discovery mode: list every region the company sells in
        host, cid = sys.argv[2], int(sys.argv[3])
        regions = list_all_regions(host, cid)
        info = {"regions": regions, "window": WINDOW.replace("DATE ", "").strip("'")}
        if not regions:  # diagnose WHY, so the UI can explain instead of just "none"
            try:
                _, rr = run_host(host, f"""SELECT count(*), cast(max(createdat) as varchar)
                    FROM transactiondb.dbo.secondarysales WHERE companyid={cid}""", 90)
                info["all_time_sales"] = int(rr[0][0] or 0)
                info["latest_sale"] = (rr[0][1] or "")[:10] or None
            except Exception:
                pass
        print("OK " + json.dumps(info))
    else:
        argv = sys.argv[1:]
        with_images = "--images" in argv
        argv = [a for a in argv if a != "--images"]
        host, cid, name, outpath = argv[0], int(argv[1]), argv[2], argv[3]
        # optional 5th arg: comma-separated regions to pull (else auto goldilocks top-3)
        regions_arg = argv[4].split(",") if len(argv) > 4 and argv[4] else None
        frame, regions, n_sell = pull_one(host, cid, name, regions_arg, with_images=with_images)
        frame.write_parquet(outpath)
        print("OK " + json.dumps({"regions": regions, "selling": n_sell}))
