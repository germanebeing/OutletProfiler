"""Segmentation engine — the hypothesis under test.

Pipeline: type (clean format/channel) -> structural peer group
(channel x format x density-band) -> per-lever frontier (p80 within peer) ->
realisation index -> tier -> decorrelation guard -> cross-company baseline for
no-data outlets -> action-projection grades.

THE CORE LESSON (validated the hard way): a peer-frontier built on *extensive
counts* (raw distinct_skus, raw bills) just rebuilds the size grade — RI
correlated 0.66 with log(throughput) and the guard failed. Controllables must
be size-neutral INTENSITY RATIOS:
  - range_intensity = distinct SKUs per bill   (basket breadth, not basket size)
  - cadence         = share of weeks with an order   (consistency, not volume)
  - recency         = days since last bill    (freshness; size-neutral by nature)
basket_value (₹/bill) is computed but HELD OUT of the index — it leaks size and
is used only as a premium-capacity proxy in action projection, clearly flagged.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import polars as pl

TODAY = _dt.date(2026, 7, 1)
WINDOW_WEEKS = 13          # Apr 1 -> Jul 1 window
MIN_PEER = 20              # below this cell size, coarsen the peer group
# Per-company peer-count caps, by how the company was segmented (seg_mode):
#   text  → structural attributes only: too many segments (18, 60) aren't
#           interpretable and dilute the frontier, so cap hard at 6.
#   both  → image classification + attributes: richer, allow more but still club
#           an over-fragmented company down to a readable number.
#   image → the image classification IS the segmentation; keep every type, no cap.
MAX_TEXT_PEERS = 6
MAX_BOTH_PEERS = 12
_MODE_CAPS = {"text": MAX_TEXT_PEERS, "both": MAX_BOTH_PEERS}  # image → None (uncapped)
_OTHER_PEER = "mixed / other"
MIN_MATURE = 10            # a peer needs this many mature outlets to set its own frontier
MIN_BILLS = 2              # maturity gate: fewer than this = too thin to grade confidently
MIN_WEEKS = 2              # maturity gate: ordered in fewer weeks = too thin
FRONTIER_PCTL = 0.80       # the demonstrated ceiling
GUARD_MAX = 0.5            # |Spearman(lever, log size)| must stay under this
VALUE_WEIGHT = 0.20        # client directive: total sales value gets SOME weight
                           # (peer-normalised value vs peer p80), not zero. Raises
                           # the size-correlation ~linearly with weight — reported.

# The scored controllables — all size-neutral intensities. (column, direction)
CONTROLLABLES = [("range_intensity", "higher"), ("cadence", "higher"), ("recency", "lower")]
# Diagnostics computed but NOT scored (size-leaky / outcome-ish).
DIAGNOSTIC_LEVERS = ["basket_value"]


# ─── typing (clean the dirty free-text; image is the production upgrade) ──

def _clean_format(s: str | None) -> str:
    t = (s or "").strip().lower()
    if not t:
        return "unknown"
    def has(*ks): return any(k in t for k in ks)
    if has("chemist", "pharma", "medical", "druggist"): return "chemist"
    if has("pan ", "paan", "kiosk", "gutka", "beedi", "tobacco"): return "pan_kiosk"
    if has("super", "mart", "modern", "hyper", "mall", "reliance", "dmart"): return "supermarket"
    if has("cosmetic", "salon", "beauty"): return "cosmetics"
    if has("hotel", "restaurant", "horeca", "dhaba", "eatery", "cafe", "bakery", "sweet", "confection"): return "horeca"
    if has("wholesale", "distributor", "stockist"): return "wholesale"
    if has("kirana", "grocer", "general", "provision", "retail", "store", "shop"): return "kirana"
    return "other"


def _clean_channel(s: str | None) -> str:
    t = (s or "").strip().lower()
    if not t: return "unknown"
    if "horeca" in t or "qsr" in t or "restaurant" in t: return "HoReCa"
    if t == "mt" or "modern" in t: return "MT"
    if "gt" in t or "general" in t or "grocer" in t: return "GT"
    return "GT"  # emerging-market base is GT (Fusion/SRA/Pharmacy/CSD are GT routes)


def _format_from_channel(s: str | None) -> str:
    """Fallback typing when shoptype text is absent (e.g. Colgate populates the
    channel, not the free-text shop type). Production uses the image; this is the
    validation stand-in."""
    t = (s or "").strip().lower()
    if not t: return "unknown"
    if "pharmac" in t or "chemist" in t or "medical" in t: return "chemist"
    if "qsr" in t or "restaurant" in t or "food" in t or "horeca" in t or "hotel" in t: return "horeca"
    if "super" in t or "modern" in t: return "supermarket"
    if "csd" in t or "canteen" in t or "institut" in t or "work" in t: return "other"
    if "gt" in t or "general" in t or "grocer" in t: return "kirana"
    return "unknown"


# ─── engine ──────────────────────────────────────────────────────────────

@dataclass
class Result:
    graded: pl.DataFrame          # every outlet with peer, RI, tier, gaps
    peers: pl.DataFrame           # peer-group summary (frontiers, sizes)
    baseline: pl.DataFrame        # cross-company network baseline per peer-tuple
    validation: dict              # the hypothesis-test metrics


def _controllables(df: pl.DataFrame) -> pl.DataFrame:
    # cast to Utf8 first so an all-null (Null-typed) last_bill column doesn't
    # crash .str; strip fractional seconds so both Trino (…10.384) and ClickHouse
    # (…00.000) timestamps parse under one format.
    last = (pl.col("last_bill").cast(pl.Utf8, strict=False)
            .str.replace(r"\.\d+$", "").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))
    bills = pl.col("bills").cast(pl.Float64)
    weeks = pl.col("order_weeks").cast(pl.Float64)
    ok = pl.col("has_data") & (bills > 0)
    return df.with_columns(
        # type off the shop-type text; where it's absent, fall back to the channel
        format=pl.when(pl.col("shoptypename").map_elements(_clean_format, return_dtype=pl.Utf8, skip_nulls=False) != "unknown")
        .then(pl.col("shoptypename").map_elements(_clean_format, return_dtype=pl.Utf8, skip_nulls=False))
        .otherwise(pl.col("channelname").map_elements(_format_from_channel, return_dtype=pl.Utf8, skip_nulls=False)),
        channel=pl.col("channelname").map_elements(_clean_channel, return_dtype=pl.Utf8, skip_nulls=False),
        range_intensity=pl.when(ok).then(pl.col("distinct_skus").cast(pl.Float64) / bills).otherwise(None),
        cadence=pl.when(ok).then(pl.min_horizontal(pl.lit(1.0), weeks / WINDOW_WEEKS)).otherwise(None),
        basket_value=pl.when(ok).then(pl.col("line_value").cast(pl.Float64) / bills).otherwise(None),
        recency=pl.when(ok).then((pl.lit(TODAY).cast(pl.Datetime) - last).dt.total_days().cast(pl.Float64)).otherwise(None),
        # maturity gate: too few bills / too few distinct order-weeks = too thin
        # to grade with confidence (a 1-bill outlet trivially maxes an intensity).
        mature=pl.col("has_data") & (bills >= MIN_BILLS) & (weeks >= MIN_WEEKS),
    )


def _density_band(df: pl.DataFrame) -> pl.DataFrame:
    # density = registered outlets per beat, within company (structural, size-blind).
    dens = df.group_by(["company_id", "beatid"]).len().rename({"len": "beat_outlets"})
    df = df.join(dens, on=["company_id", "beatid"], how="left")
    q = df.group_by("company_id").agg(
        pl.col("beat_outlets").quantile(0.33).alias("d1"),
        pl.col("beat_outlets").quantile(0.66).alias("d2"),
    )
    df = df.join(q, on="company_id", how="left").with_columns(
        pl.when(pl.col("beat_outlets") <= pl.col("d1")).then(pl.lit("sparse"))
        .when(pl.col("beat_outlets") <= pl.col("d2")).then(pl.lit("medium"))
        .otherwise(pl.lit("dense")).alias("density")
    ).drop(["d1", "d2"])
    return df


AXIS_COL = {"channel": "channel", "format": "format", "region": "regionname",
            "affluence": "affluence_tier", "density": "density"}
DEFAULT_AXES = ["channel", "format", "region"]   # evidence-based default (see factor_validate.py)


def _assign_peers(df: pl.DataFrame, axes: list[str] | None = None) -> pl.DataFrame:
    # The segment/peer is a business-adjustable choice of structural axes. Default
    # is channel·format·region (best per-company guard); the Segments UI can swap
    # in affluence/density and re-segment. Coarsens to channel·format when a cell
    # is too small to set a frontier.
    axes = [a for a in (axes or DEFAULT_AXES) if a in AXIS_COL] or DEFAULT_AXES
    cols = [pl.col(AXIS_COL[a]).cast(pl.Utf8).fill_null("?") for a in axes]
    full = cols[0]
    for c in cols[1:]:
        full = full + "·" + c
    df = df.with_columns(peer_full=full)
    sizes = df.filter(pl.col("has_data")).group_by(["company_id", "peer_full"]).len()
    small = {(r["company_id"], r["peer_full"])
             for r in sizes.filter(pl.col("len") < MIN_PEER).iter_rows(named=True)}

    def resolve(row):
        if (row["company_id"], row["peer_full"]) in small:      # coarsen to channel·format
            return (row["channel"] or "unknown") + "·" + (row["format"] or "unknown")
        return row["peer_full"]

    df = df.with_columns(
        peer=pl.struct(["company_id", "peer_full", "channel", "format"])
              .map_elements(resolve, return_dtype=pl.Utf8)
    )
    # mode-aware per-company cap. A company that over-fragments produces tiny peer
    # groups no one can reason about; how hard we club depends on how it was
    # segmented (seg_mode: text / image / both — see _MODE_CAPS). Text is capped
    # hardest; image keeps every classified type.
    smode = (pl.col("seg_mode") if "seg_mode" in df.columns else pl.lit("text")).fill_null("text")
    df = df.with_columns(_seg_mode=smode)
    cf = pl.col("channel").fill_null("unknown") + "·" + pl.col("format").fill_null("unknown")
    cmode = (df.filter(pl.col("has_data")).group_by("company_id")
             .agg(pl.col("_seg_mode").first().alias("mode"),
                  pl.col("peer").n_unique().alias("np")))
    for r in cmode.iter_rows(named=True):
        cap = _MODE_CAPS.get(r["mode"])
        if cap is None or r["np"] <= cap:
            continue
        cid = r["company_id"]
        is_co = pl.col("company_id") == cid
        # step 1: coarsen this company's whole segmentation to channel·format
        df = df.with_columns(peer=pl.when(is_co).then(cf).otherwise(pl.col("peer")))
        # step 2: if still over cap, keep the (cap-1) largest peers and bucket the
        # rest into one readable "mixed / other" group — guarantees ≤ cap groups.
        order = (df.filter(is_co & pl.col("has_data")).group_by("peer").len()
                 .sort("len", descending=True).get_column("peer").to_list())
        if len(order) > cap:
            keepset = order[:cap - 1]
            df = df.with_columns(peer=pl.when(is_co & ~pl.col("peer").is_in(keepset))
                                 .then(pl.lit(_OTHER_PEER)).otherwise(pl.col("peer")))
    return df.drop("_seg_mode")


def _frontier_and_tier(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # The frontier is the demonstrated ceiling of MATURE outlets — thin, 1-bill
    # outlets must not define what "good" looks like for their peer. Peers with
    # too few mature outlets fall back to the all-has_data frontier.
    def agg_frame(sub: pl.DataFrame, sfx: str) -> pl.DataFrame:
        a = [pl.len().alias(f"n{sfx}")]
        for m, direction in CONTROLLABLES:
            pct = FRONTIER_PCTL if direction == "higher" else (1 - FRONTIER_PCTL)
            a.append(pl.col(m).quantile(pct).alias(f"front_{m}{sfx}"))
        a.append(pl.col("line_value").quantile(FRONTIER_PCTL).alias(f"front_value{sfx}"))
        for m in DIAGNOSTIC_LEVERS:
            a.append(pl.col(m).median().alias(f"med_{m}{sfx}"))
        return sub.group_by(["company_id", "peer"]).agg(*a)

    allp = agg_frame(df.filter(pl.col("has_data")), "")       # n, front_*, med_*
    mat = agg_frame(df.filter(pl.col("mature")), "_m")        # n_m, front_*_m
    peers = allp.join(mat, on=["company_id", "peer"], how="left").with_columns(
        pl.col("n_m").fill_null(0))
    for m in [c for c, _ in CONTROLLABLES] + ["value"]:        # prefer mature frontier
        peers = peers.with_columns(
            pl.when(pl.col("n_m") >= MIN_MATURE).then(pl.col(f"front_{m}_m"))
            .otherwise(pl.col(f"front_{m}")).alias(f"front_{m}"))
    peers = peers.rename({"n": "peer_n", "n_m": "mature_n"}).select(
        ["company_id", "peer", "peer_n", "mature_n", "front_value"]
        + [f"front_{m}" for m, _ in CONTROLLABLES] + [f"med_{m}" for m in DIAGNOSTIC_LEVERS])
    df = df.join(peers, on=["company_id", "peer"], how="left")

    real_cols = []
    for m, direction in CONTROLLABLES:
        f = pl.col(f"front_{m}")
        if direction == "higher":
            r = pl.min_horizontal(pl.lit(1.0), pl.col(m) / f)
        else:                                        # lower better: realisation = frontier/value
            r = pl.min_horizontal(pl.lit(1.0), f / pl.col(m))
        df = df.with_columns(r.alias(f"real_{m}"))
        real_cols.append(f"real_{m}")

    # value realisation: total sales value vs the peer's p80 value ceiling (higher better)
    df = df.with_columns(
        real_value=pl.min_horizontal(pl.lit(1.0), pl.col("line_value") / pl.col("front_value")))
    # RI = size-neutral intensities (weight 1-w) blended with value realisation (weight w).
    # RI_intensity kept alongside so the effect of value is visible/removable.
    df = df.with_columns(
        RI_intensity=pl.mean_horizontal([pl.col(c) for c in real_cols]))
    df = df.with_columns(
        RI=(1 - VALUE_WEIGHT) * pl.col("RI_intensity") + VALUE_WEIGHT * pl.col("real_value"))
    tier = (pl.when(pl.col("RI") >= 0.85).then(pl.lit("T1"))
            .when(pl.col("RI") >= 0.65).then(pl.lit("T2"))
            .when(pl.col("RI") >= 0.45).then(pl.lit("T3"))
            .otherwise(pl.lit("T4")))
    df = df.with_columns(
        pl.when(pl.col("has_data")).then(tier).otherwise(pl.lit("provisional")).alias("tier"),
        gap=(1 - pl.col("RI")),
        confidence=pl.when(~pl.col("has_data")).then(pl.lit("none"))
        .when(~pl.col("mature")).then(pl.lit("low"))            # too thin to trust the tier
        .when(pl.col("format") == "unknown").then(pl.lit("low"))  # needs image typing
        .otherwise(pl.lit("high")),
    )
    return df, peers


def _baseline(df: pl.DataFrame) -> pl.DataFrame:
    # cross-company network baseline per standard peer TUPLE (channel·format·density):
    # median intensities + how many companies contribute. Grades no-data outlets.
    data = df.filter(pl.col("has_data"))
    # cross-company axes must be company-agnostic: channel + format (region names
    # differ per company; density was dropped as noise). This is the cold-start key.
    tup = (pl.col("channel") + "·" + pl.col("format"))
    b = data.with_columns(tuple_key=tup).group_by("tuple_key").agg(
        companies=pl.col("company_id").n_unique(),
        company_ids=pl.col("company_id").unique(),   # keep the set, so cold-start can
        n=pl.len(),                                  # exclude an outlet's OWN company
        front_range_intensity=pl.col("range_intensity").quantile(0.8),
        front_cadence=pl.col("cadence").quantile(0.8),
        med_basket_value=pl.col("basket_value").median(),
    )
    return b.sort("n", descending=True)


def _spearman(a: pl.Series, b: pl.Series) -> float:
    m = a.is_not_null() & b.is_not_null()
    a, b = a.filter(m), b.filter(m)
    if a.len() < 10 or a.n_unique() < 2 or b.n_unique() < 2:
        return float("nan")
    val = pl.DataFrame({"a": a.rank(), "b": b.rank()}).select(pl.corr("a", "b")).item()
    return float(val) if val is not None else float("nan")


def _validation(df: pl.DataFrame) -> dict:
    # throughput guard uses line_value (qty*ptr from line items): the header
    # invoiceamount is 0 for ~54% of bills, line_value is the real ₹ throughput.
    # Guard runs on MATURE outlets — the population we grade with confidence.
    data = df.filter(pl.col("mature") & pl.col("RI").is_not_null() & (pl.col("line_value") > 0))
    if data.height < 30:
        return {"note": "insufficient data for guard"}
    logv = data.select(pl.col("line_value").log1p()).to_series()
    ri = data.select("RI").to_series()

    per_lever = {m: round(_spearman(data.select(f"real_{m}").to_series(), logv), 3)
                 for m, _ in CONTROLLABLES}
    # value-weight trade-off: the client wants total sales value to count. Show how
    # much size-correlation each weight buys, so the weight is a transparent dial.
    ri_int = data.select("RI_intensity").to_series()
    rv = data.select("real_value").to_series()
    value_weight_curve = {}
    for w in (0.0, 0.2, 0.35, 0.5):
        blended = (1 - w) * ri_int + w * rv
        value_weight_curve[w] = round(_spearman(blended, logv), 3)
    decorr_intensity_only = round(_spearman(ri_int, logv), 3)
    # also show what the leaky diagnostic would have scored, for contrast
    diag = {m: round(_spearman(data.select(m).to_series(), logv), 3) for m in DIAGNOSTIC_LEVERS}
    # what the RAW extensive counts score (the trap we avoided)
    raw = {
        "raw_distinct_skus": round(_spearman(data.select("distinct_skus").to_series(), logv), 3),
        "raw_bills": round(_spearman(data.select("bills").to_series(), logv), 3),
    }
    ri_corr = round(_spearman(ri, logv), 3)
    tier_dist = df.filter(pl.col("has_data")).group_by("tier").len().sort("tier").to_dict(as_series=False)

    # returns quality signal (Colgate): is return_rate size-neutral, and how much
    # value flows back? Reported, not scored into RI.
    ret = df.filter(pl.col("has_returns").fill_null(False)
                    & pl.col("has_data") & (pl.col("line_value") > 0)) \
        if "has_returns" in df.columns else df.head(0)
    returns_block = None
    if ret.height >= 30:
        rlogv = ret.select(pl.col("line_value").log1p()).to_series()
        rr_corr = round(_spearman(ret.select("return_rate").to_series(), rlogv), 3)
        returns_block = {
            "companies_with_returns": ret.select(pl.col("company_name").unique()).to_series().to_list(),
            "n_outlets": ret.height,
            "mean_return_rate": round(ret.select(pl.col("return_rate").mean()).item() or 0, 3),
            "pct_outlets_with_any_return": round(
                100 * ret.select((pl.col("return_bills").cast(pl.Float64, strict=False) > 0).mean()).item(), 1),
            # honest: this is a REAL positive correlation (bigger outlets return a
            # bit more), just under the 0.5 guard. Not "size-neutral" — reported,
            # not scored into RI.
            "return_rate_vs_logsize": rr_corr,
            "return_rate_size_correlation": "weak" if abs(rr_corr) < 0.35 else "moderate",
        }

    # identical-sales falsifier: same-peer pairs within ±5% throughput that land in different tiers
    same = _identical_sales_test(data)

    # PER-COMPANY decorrelation — the pooled number is a cross-company average and
    # can hide a single-company throughput leak. This is how the grade actually
    # ships (one company at a time), so it must be surfaced, not averaged away.
    per_company = {}
    for co in data.select("company_name").unique().to_series().to_list():
        s = data.filter(pl.col("company_name") == co)
        if s.height >= 50:
            per_company[co] = round(_spearman(
                s.select("RI").to_series(), s.select(pl.col("line_value").log1p()).to_series()), 3)
    companies_over_guard = sorted([c for c, v in per_company.items() if abs(v) > GUARD_MAX])

    # FULL has_data population (not just mature): exposes how much the mature-only
    # guard pass depends on the maturity restriction (cadence is the fragile lever).
    full = df.filter(pl.col("has_data") & pl.col("RI").is_not_null() & (pl.col("line_value") > 0))
    flog = full.select(pl.col("line_value").log1p()).to_series()
    per_lever_full = {m: round(_spearman(full.select(f"real_{m}").to_series(), flog), 3)
                      for m, _ in CONTROLLABLES}

    return {
        "n_graded": data.height,
        "decorrelation_RI_vs_logsize": ri_corr,
        "guard_pass": abs(ri_corr) <= GUARD_MAX and all(
            (v != v) or abs(v) <= GUARD_MAX for v in per_lever.values()),
        "guard_pass_every_company": len(companies_over_guard) == 0,
        "per_company_decorrelation": per_company,
        "companies_over_guard": companies_over_guard,
        "per_lever_vs_logsize": per_lever,
        "per_lever_full_population (maturity-dependence)": per_lever_full,
        "value_weight_used": VALUE_WEIGHT,
        "decorrelation_intensity_only (value weight 0)": decorr_intensity_only,
        "decorrelation_by_value_weight": {str(k): v for k, v in value_weight_curve.items()},
        "diagnostic_levers_vs_logsize": diag,
        "raw_count_levers_vs_logsize (the trap)": raw,
        "identical_sales_divergence_pct": same,
        "tier_distribution": dict(zip(tier_dist.get("tier", []), tier_dist.get("len", []))),
        "n_peers": df.select(pl.col("peer").n_unique()).item(),
        "n_nodata": df.filter(~pl.col("has_data")).height,
        "n_mature": int(df.filter(pl.col("mature")).height),
        "n_immature_thin": int(df.filter(pl.col("has_data") & ~pl.col("mature")).height),
        "returns": returns_block,
    }


def _identical_sales_test(data: pl.DataFrame) -> float:
    """Within a peer, take outlets close in throughput (±5%); what share of
    adjacent pairs land in DIFFERENT tiers? High = the grade sees past size."""
    d = data.select(["peer", "line_value", "tier"]).sort(["peer", "line_value"])
    rows = d.iter_rows(named=True)
    prev = None
    pairs = diverge = 0
    for r in rows:
        if prev and prev["peer"] == r["peer"] and prev["line_value"] > 0:
            if abs(r["line_value"] - prev["line_value"]) / prev["line_value"] <= 0.05:
                pairs += 1
                if r["tier"] != prev["tier"]:
                    diverge += 1
        prev = r
    return round(100 * diverge / pairs, 1) if pairs else 0.0


def _returns(df: pl.DataFrame) -> pl.DataFrame:
    """Fold in returns (NonFA InvoiceType 1/2, Colgate). return_rate is a
    size-neutral QUALITY ratio: share of gross value that flowed back. Surfaced
    signal + action penalty — NOT a scored frontier lever, so RI stays comparable
    across companies without returns coverage."""
    for c in ("returns_value", "return_bills"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    df = df.with_columns(
        pl.col("returns_value").cast(pl.Float64, strict=False),
        pl.col("return_bills").cast(pl.Float64, strict=False),
    )
    has_ret = df.group_by("company_id").agg(
        (pl.col("returns_value").fill_null(0).abs().sum() > 0).alias("has_returns"))
    df = df.join(has_ret, on="company_id", how="left")
    gross = pl.col("line_value") + pl.col("returns_value")
    return df.with_columns(
        return_rate=pl.when(pl.col("has_returns") & pl.col("has_data") & (gross > 0))
        .then(pl.min_horizontal(pl.lit(1.0), pl.col("returns_value") / gross))
        .otherwise(None))


def run_engine(parquet: str, peer_axes: list[str] | None = None) -> Result:
    df = pl.read_parquet(parquet)
    df = df.with_columns(
        pl.col("total_value").cast(pl.Float64, strict=False),
        pl.col("line_value").cast(pl.Float64, strict=False),
        pl.col("bills").cast(pl.Float64, strict=False),
        pl.col("distinct_skus").cast(pl.Float64, strict=False),
        pl.col("order_weeks").cast(pl.Float64, strict=False),
    )
    if "affluence_tier" not in df.columns:      # synthetic/older data -> single tier
        df = df.with_columns(affluence_tier=pl.lit("na"))
    df = _returns(df)
    df = _controllables(df)
    df = _density_band(df)
    df = _assign_peers(df, peer_axes)
    df, peers = _frontier_and_tier(df)
    baseline = _baseline(df)
    validation = _validation(df)
    keep = ["company_id", "company_name", "outletid", "regionname", "city", "beatid",
            "format", "channel", "density", "affluence_tier", "peer", "peer_n", "mature_n", "has_data", "mature", "confidence",
            "bills", "total_value", "line_value", "distinct_skus", "order_weeks", "segmentationname",
            "range_intensity", "cadence", "recency", "basket_value",
            "returns_value", "return_bills", "return_rate", "has_returns",
            "real_range_intensity", "real_cadence", "real_recency", "real_value",
            "RI_intensity", "RI", "gap", "tier"]
    graded = df.select([c for c in keep if c in df.columns])
    return Result(graded=graded, peers=peers, baseline=baseline, validation=validation)


if __name__ == "__main__":
    import json
    import os
    parquet = next((p for p in ("data/outlets_geo2.parquet", "data/outlets_geo.parquet", "data/outlets_all.parquet",
                                 "data/outlets.parquet") if os.path.exists(p)), "data/outlets.parquet")
    res = run_engine(parquet)
    res.graded.write_parquet("data/graded.parquet")
    res.peers.write_parquet("data/peers.parquet")
    res.baseline.write_parquet("data/baseline.parquet")
    print(json.dumps(res.validation, indent=2, default=str))
