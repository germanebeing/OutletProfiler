"""Affluence axis from SCRAPED external data (GeoNames, free/open).

pincode -> place/district (GeoNames IN postal), place -> population (GeoNames
cities5000). Population is a defensible urbanisation/affluence proxy: metros skew
affluent, rural towns less so. Real SEC/NCCS is licensed; this is the open-data
stand-in the client asked to scrape.

Writes data/outlets_geo2.parquet (+ affluence_tier, district) and validates
affluence_tier as a peer factor (eta^2 on behaviour vs on size).
"""
from __future__ import annotations

import zipfile

import polars as pl

SCR = "/private/tmp/claude-501/-Users-prithvi-classifieragent/e9b1580e-beab-499b-b1dd-80d9e81cd77b/scratchpad"

POSTAL_COLS = ["cc", "pincode", "place", "state", "sc1", "district", "sc2", "adm3", "sc3", "plat", "plon", "acc"]
CITY_COLS = ["gid", "name", "ascii", "alt", "clat", "clon", "fclass", "fcode", "cc", "cc2",
             "a1", "a2", "a3", "a4", "population", "elev", "dem", "tz", "mod"]


def _norm(col: pl.Expr) -> pl.Expr:
    return col.str.to_lowercase().str.strip_chars().str.replace_all(r"[^a-z ]", "")


def load_geonames():
    with zipfile.ZipFile(f"{SCR}/IN_zip.zip") as z:
        postal = pl.read_csv(z.read("IN.txt"), separator="\t", has_header=False,
                             new_columns=POSTAL_COLS, schema_overrides={"pincode": pl.Utf8},
                             infer_schema_length=0)
    with zipfile.ZipFile(f"{SCR}/IN_cities5000.zip") as z:
        cities = pl.read_csv(z.read("cities5000.txt"), separator="\t", has_header=False,
                             new_columns=CITY_COLS, infer_schema_length=0)
    cities = cities.filter(pl.col("cc") == "IN").with_columns(
        pop=pl.col("population").cast(pl.Int64, strict=False), key=_norm(pl.col("ascii")))
    citypop = (cities.group_by("key").agg(pl.col("pop").max())
               .filter(pl.col("pop") > 0))
    pin = postal.select("pincode", "place", "district", "state").unique(subset="pincode")
    return pin, citypop, cities


def _tier_from_pop(pop_col: pl.Expr) -> pl.Expr:
    return (pl.when(pop_col >= 5_000_000).then(pl.lit("metro"))
            .when(pop_col >= 1_000_000).then(pl.lit("tier1"))
            .when(pop_col >= 300_000).then(pl.lit("tier2"))
            .when(pop_col >= 100_000).then(pl.lit("tier3"))
            .otherwise(pl.lit("town")))


def main() -> None:
    pin, citypop, _ = load_geonames()
    df = pl.read_parquet("data/outlets_geo.parquet")

    # pincode -> place/district (external), then place & own-city -> population
    df = df.with_columns(pincode=pl.col("pincode").cast(pl.Utf8, strict=False))
    df = df.join(pin, on="pincode", how="left")
    df = df.with_columns(
        _kplace=_norm(pl.col("place").fill_null("")),
        _kcity=_norm(pl.col("city").fill_null("")))
    df = (df.join(citypop.rename({"key": "_kplace", "pop": "pop_place"}), on="_kplace", how="left")
            .join(citypop.rename({"key": "_kcity", "pop": "pop_city"}), on="_kcity", how="left"))
    df = df.with_columns(
        pop=pl.max_horizontal(pl.col("pop_place").fill_null(0), pl.col("pop_city").fill_null(0)))
    df = df.with_columns(affluence_tier=_tier_from_pop(pl.col("pop")),
                         district=pl.col("district"))
    df = df.drop(["_kplace", "_kcity", "pop_place", "pop_city"])
    df.write_parquet("data/outlets_geo2.parquet")

    tot = df.height
    matched = df.filter(pl.col("pop") > 0).height
    print(f"outlets: {tot} | population matched: {matched} ({round(100*matched/tot)}%)")
    print("affluence_tier distribution:")
    print(df.group_by("affluence_tier").len().sort("len", descending=True))

    # validate affluence_tier as a peer factor (reuse the eta^2 logic)
    from factor_validate import build, eta2
    g = build(df).filter(pl.col("has_data") & (pl.col("bills") > 0) & (pl.col("line_value") > 0))
    g = g.with_columns(logv=pl.col("line_value").log1p())
    levers = ["range_intensity", "cadence", "recency", "basket_value"]
    print("\nfactor          behav.avg  size   groups")
    for f in ["affluence_tier", "district", "format", "channel"]:
        if f not in g.columns:
            continue
        evs = [eta2(g, f, l) for l in levers]
        behav = round(sum(evs) / len(evs), 3)
        print(f"  {f:<14} {behav:<9} {eta2(g, f, 'logv'):<6} {g.select(pl.col(f).n_unique()).item()}")


if __name__ == "__main__":
    main()
