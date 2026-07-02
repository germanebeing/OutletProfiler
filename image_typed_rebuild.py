"""Rebuild peers using IMAGE-derived shop-type (CLIP), and measure whether it
helps the guard vs the text/channel typing.

Typing priority: usable CLIP image_format -> system shoptype text -> channel
fallback. Writes data/outlets_img.parquet, runs the engine on it, and prints the
before/after decorrelation so we can see if image typing moved the needle.
"""
from __future__ import annotations

import polars as pl

from engine import segment


def build_typed() -> pl.DataFrame:
    df = pl.read_parquet("data/outlets_geo2.parquet").with_columns(pl.col("outletid").cast(pl.Int64))
    img = pl.read_parquet("data/image_types.parquet")
    usable = img.filter(pl.col("usable") & pl.col("image_format").is_not_null()) \
                .select("outletid", "image_format")
    df = df.join(usable, on="outletid", how="left")
    # override the shop-type text with the (canonical) image label where we have a
    # usable one — the engine's _clean_format maps 'kirana'/'supermarket'/... through.
    df = df.with_columns(
        shoptypename=pl.when(pl.col("image_format").is_not_null())
        .then(pl.col("image_format")).otherwise(pl.col("shoptypename")))
    return df, usable.height


def summarise(tag: str, res) -> dict:
    v = res.validation
    return {"tag": tag, "pooled": v["decorrelation_RI_vs_logsize"],
            "per_company": v["per_company_decorrelation"],
            "breaches": v["companies_over_guard"],
            "identical_sales": v["identical_sales_divergence_pct"]}


def main() -> None:
    import json
    base = segment.run_engine("data/outlets_geo2.parquet")
    print("BEFORE (text/channel typing):")
    print(json.dumps(summarise("text", base), indent=1, default=str))

    typed, n_img = build_typed()
    typed.write_parquet("data/outlets_img.parquet")
    cov = typed.filter(pl.col("image_format").is_not_null() & pl.col("has_data")).height
    tot = typed.filter(pl.col("has_data")).height
    print(f"\nimage-typed outlets: {n_img} usable ({round(100*cov/tot)}% of graded outlets)")
    print("image_format distribution:")
    print(typed.filter(pl.col("image_format").is_not_null())
          .group_by("image_format").len().sort("len", descending=True))

    aft = segment.run_engine("data/outlets_img.parquet")
    print("\nAFTER (image typing where usable):")
    print(json.dumps(summarise("image", aft), indent=1, default=str))


if __name__ == "__main__":
    main()
