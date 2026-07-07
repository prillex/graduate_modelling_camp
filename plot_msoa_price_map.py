
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / ".cache"))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "msoa_price_map"

PRICE_DATA = DATA_DIR / "Cambridge data.csv"
BOUNDARIES = DATA_DIR / "Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V3_-4477917303172606123.geojson"


def format_currency(value: float) -> str:
    return f"GBP {value:,.0f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(PRICE_DATA)
    boundaries = gpd.read_file(BOUNDARIES)

    msoa_prices = (
        prices.groupby("msoa21")["price_last"]
        .agg(row_count="count", mean_price="mean", median_price="median")
        .reset_index()
    )

    mapped = boundaries.merge(msoa_prices, left_on="MSOA21CD", right_on="msoa21", how="inner")


    mapped_projected = mapped.to_crs(epsg=27700)

    fig, ax = plt.subplots(figsize=(10, 11))
    mapped_projected.plot(
        column="median_price",
        ax=ax,
        cmap="viridis",
        linewidth=0.6,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": "Median price", "shrink": 0.65},
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    mapped_projected.boundary.plot(ax=ax, linewidth=0.25, color="#333333", alpha=0.45)

    ax.set_title("Median House Price by MSOA", fontsize=16, pad=14)
    ax.set_axis_off()

    note = (
        f"{len(mapped):,} MSOAs matched | "
        f"{len(prices):,} rows | "
        f"Median range: {format_currency(mapped['median_price'].min())} to {format_currency(mapped['median_price'].max())}"
    )
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=9, color="#333333")

    png_path = OUTPUT_DIR / "median_price_by_msoa.png"
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    mapped.drop(columns="geometry").sort_values("median_price", ascending=False).to_csv(
        OUTPUT_DIR / "msoa_price_map_data.csv",
        index=False,
    )

    geojson_path = OUTPUT_DIR / "msoa_price_map_data.geojson"
    mapped.to_file(geojson_path, driver="GeoJSON")

    print(f"Matched {len(mapped):,} MSOAs from the price data.")
    print(f"Saved {png_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {geojson_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {(OUTPUT_DIR / 'msoa_price_map_data.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
