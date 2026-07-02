"""Validate WHICH peer factors are worth grouping on, empirically.

A good peer axis = it explains real variance in how outlets behave (eta^2 on the
controllables) — shops in the same group genuinely behave alike, so the frontier
comparison is fair — WITHOUT just being a size proxy (eta^2 on log throughput
must stay low, else the peer axis smuggles size back in).

Prints, per candidate factor: eta^2 on each behavioural lever, and eta^2 on size.
"""
from __future__ import annotations

import polars as pl

from engine.segment import _clean_channel, _clean_format, _format_from_channel

WINDOW_WEEKS = 13


def eta2(df: pl.DataFrame, factor: str, y: str) -> float:
    d = df.select([factor, y]).drop_nulls()
    if d.height < 30 or d.select(pl.col(factor).n_unique()).item() < 2:
        return 0.0
    grand = d.select(pl.col(y).mean()).item()
    ss_tot = d.select(((pl.col(y) - grand) ** 2).sum()).item()
    if not ss_tot:
        return 0.0
    grp = d.group_by(factor).agg(n=pl.len(), m=pl.col(y).mean())
    ss_bet = grp.select((pl.col("n") * (pl.col("m") - grand) ** 2).sum()).item()
    return round(ss_bet / ss_tot, 3)


def build(df: pl.DataFrame) -> pl.DataFrame:
    fmt = pl.col("shoptypename").map_elements(_clean_format, return_dtype=pl.Utf8, skip_nulls=False)
    df = df.with_columns(
        channel=pl.col("channelname").map_elements(_clean_channel, return_dtype=pl.Utf8, skip_nulls=False),
        format=pl.when(fmt != "unknown").then(fmt)
        .otherwise(pl.col("channelname").map_elements(_format_from_channel, return_dtype=pl.Utf8, skip_nulls=False)),
        geo_market=pl.col("marketname"),
        geo_city=pl.col("city"),
        # behavioural levers (same defs as the engine)
        range_intensity=pl.col("distinct_skus").cast(pl.Float64) / pl.col("bills").cast(pl.Float64),
        cadence=pl.min_horizontal(1.0, pl.col("order_weeks").cast(pl.Float64) / WINDOW_WEEKS),
        basket_value=pl.col("line_value").cast(pl.Float64) / pl.col("bills").cast(pl.Float64),
        recency=(pl.lit(__import__("datetime").date(2026, 7, 1)).cast(pl.Datetime)
                 - pl.col("last_bill").cast(pl.Utf8, strict=False).str.replace(r"\.\d+$", "")
                 .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)).dt.total_days().cast(pl.Float64),
    )
    # GPS catchment density: outlets in the same ~1.1km grid cell (within company)
    df = df.with_columns(
        cell=(pl.col("lat").round(2).cast(pl.Utf8) + "_" + pl.col("lon").round(2).cast(pl.Utf8)))
    dens = df.group_by(["company_id", "cell"]).agg(pl.len().alias("cell_n"))
    df = df.join(dens, on=["company_id", "cell"], how="left")
    q = df.group_by("company_id").agg(d1=pl.col("cell_n").quantile(0.33), d2=pl.col("cell_n").quantile(0.66))
    df = df.join(q, on="company_id", how="left").with_columns(
        density_gps=pl.when(pl.col("lat").is_null()).then(None)
        .when(pl.col("cell_n") <= pl.col("d1")).then(pl.lit("sparse"))
        .when(pl.col("cell_n") <= pl.col("d2")).then(pl.lit("medium")).otherwise(pl.lit("dense")))
    # old beat-count density, for comparison
    b = df.group_by(["company_id", "beatid"]).agg(pl.len().alias("beat_n"))
    df = df.join(b, on=["company_id", "beatid"], how="left")
    qb = df.group_by("company_id").agg(b1=pl.col("beat_n").quantile(0.33), b2=pl.col("beat_n").quantile(0.66))
    df = df.join(qb, on="company_id", how="left").with_columns(
        density_beat=pl.when(pl.col("beat_n") <= pl.col("b1")).then(pl.lit("sparse"))
        .when(pl.col("beat_n") <= pl.col("b2")).then(pl.lit("medium")).otherwise(pl.lit("dense")))
    # pincode as a fine geo/affluence proxy (Trino cos only)
    df = df.with_columns(pin3=pl.col("pincode").str.slice(0, 3))
    return df


def main() -> None:
    df = pl.read_parquet("data/outlets_geo.parquet").with_columns(
        pl.col("bills").cast(pl.Float64, strict=False),
        pl.col("line_value").cast(pl.Float64, strict=False),
        pl.col("distinct_skus").cast(pl.Float64, strict=False),
        pl.col("order_weeks").cast(pl.Float64, strict=False),
    )
    df = build(df).filter(pl.col("has_data") & (pl.col("bills") > 0) & (pl.col("line_value") > 0))
    df = df.with_columns(logv=pl.col("line_value").log1p())

    factors = ["channel", "format", "geo_market", "geo_city", "pin3",
               "density_gps", "density_beat"]
    levers = ["range_intensity", "cadence", "recency", "basket_value"]

    print(f"n = {df.height} graded outlets\n")
    print(f"{'factor':<14} {'behav.avg':>9} | " + " ".join(f'{l[:8]:>8}' for l in levers) + f" | {'SIZE':>6} {'groups':>7}")
    print("-" * 88)
    rows = []
    for f in factors:
        evs = [eta2(df, f, l) for l in levers]
        size = eta2(df, f, "logv")
        behav = round(sum(evs) / len(evs), 3)
        ng = df.select(pl.col(f).n_unique()).item()
        rows.append((f, behav, size))
        print(f"{f:<14} {behav:>9} | " + " ".join(f'{e:>8}' for e in evs) + f" | {size:>6} {ng:>7}")
    print("\nRead: higher 'behav.avg' = the factor genuinely separates how outlets behave (good peer axis).")
    print("      'SIZE' should stay LOW — a high value means the factor is smuggling size back in.")
    print("\nRanking (best peer axes = high behaviour, low size):")
    for f, b, s in sorted(rows, key=lambda r: r[1] - r[2], reverse=True):
        verdict = "STRONG" if b >= 0.05 and s < 0.15 else ("size-ish" if s >= 0.15 else "weak")
        print(f"  {f:<14} behaviour={b:<6} size={s:<6} -> {verdict}")


if __name__ == "__main__":
    main()
