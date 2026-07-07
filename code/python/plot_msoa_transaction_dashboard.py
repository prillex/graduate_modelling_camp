"""Cambridgeshire property-sales dashboard.

Builds two figures from the cleaned sales data and the MSOA 2021 boundaries:

  1. cambridgeshire_msoa_dashboard.png  - a four-panel overview: a choropleth of
     transactions per MSOA, the top MSOAs ranked, the area traits that mark a
     high-volume MSOA, and the property-type allocation of the busiest MSOAs.
  2. cambridgeshire_transaction_map.png - a standalone high-resolution version of
     the choropleth for closer inspection.

Colour follows the job it does: a single-hue blue ramp for magnitude (counts),
a fixed-order categorical set for property types, and a blue<->red diverging
scale for the trait correlations.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "image"

CLEANED = DATA_DIR / "cleaned" / "Cambridge data_cleaned.csv"
BOUNDARIES = (
    DATA_DIR
    / "spatial"
    / "Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V3_-4477917303172606123.geojson"
)

# --- design tokens (from the data-viz reference palette) ------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
POS = "#2a78d6"  # blue  - "more transactions"
NEG = "#e34948"  # red   - "fewer transactions"

# single-hue sequential ramp for counts (light -> dark blue)
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "cam_blue",
    ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

# fixed-order categorical slots for the six property groups
PROP_ORDER = ["Detached", "Semi-detached", "Terraced", "Flat/Apartment", "Bungalow", "Other"]
PROP_COLOR = {
    "Detached": "#2a78d6",
    "Semi-detached": "#1baf7a",
    "Terraced": "#eda100",
    "Flat/Apartment": "#4a3aa7",
    "Bungalow": "#eb6834",
    "Other": "#898781",
}

# 24 raw property types -> 6 display groups
PROP_GROUP = {
    "Detached house": "Detached", "Barn conversion": "Detached", "Country house": "Detached",
    "Farmhouse": "Detached", "Chalet": "Detached", "Mews house": "Detached",
    "Equestrian property": "Detached",
    "Semi-detached house": "Semi-detached", "Link-detached house": "Semi-detached",
    "Terraced house": "Terraced", "End terrace house": "Terraced", "Town house": "Terraced",
    "Cottage": "Terraced",
    "Detached bungalow": "Bungalow", "Bungalow": "Bungalow",
    "Semi-detached bungalow": "Bungalow", "Terraced bungalow": "Bungalow",
    "Flat": "Flat/Apartment", "Maisonette": "Flat/Apartment", "Studio": "Flat/Apartment",
    "Mobile/park home": "Other", "Lodge": "Other", "Houseboat": "Other", "Parking/garage": "Other",
}

# area traits to correlate with transaction volume, with reader-facing labels
TRAITS = [
    ("median_price", "Higher median price"),
    ("Work from Home", "More work-from-home"),
    ("Cycle", "More cycle commuting"),
    ("Foot", "More walk commuting"),
    ("0-5km", "Lives near work (<5km)"),
    ("High", "Higher qualifications"),
    ("Low", "Lower qualifications"),
    ("Minors 0-18", "More children (0-18)"),
    ("Driving", "More car commuting"),
]


def style_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def load() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    df = pd.read_csv(CLEANED)
    df["grp"] = df["property_type"].map(PROP_GROUP)

    demo = ["median_price", "Work from Home", "Cycle", "Foot", "0-5km", "High", "Low",
            "Minors 0-18", "Driving"]
    per = df.groupby("msoa21").agg(
        count=("price_sold", "size"),
        median_price=("price_sold", "median"),
        **{c: (c, "first") for c in demo if c != "median_price"},
    )

    gdf = gpd.read_file(BOUNDARIES, columns=["MSOA21CD", "MSOA21NM", "geometry"])
    gdf = gdf[gdf["MSOA21CD"].isin(per.index)].copy()
    gdf = gdf.merge(per, left_on="MSOA21CD", right_index=True).to_crs(epsg=27700)
    gdf["short"] = gdf["MSOA21NM"].str.replace("Cambridge ", "C", regex=False)
    return df, gdf


def draw_map(ax, gdf, *, standalone=False) -> None:
    vmax = gdf["count"].max()
    gdf.plot(ax=ax, column="count", cmap=BLUE_RAMP, norm=Normalize(0, vmax),
             linewidth=0.6, edgecolor="white")
    ax.set_axis_off()
    ax.set_aspect("equal")

    n_lab = 8 if standalone else 6
    top = gdf.sort_values("count", ascending=False).head(n_lab)
    for _, row in top.iterrows():
        pt = row.geometry.representative_point()
        ax.annotate(
            f"{row['short']}\n{int(row['count']):,}",
            (pt.x, pt.y), ha="center", va="center",
            fontsize=8.5 if standalone else 7.5, fontweight="bold", color="white",
            linespacing=1.05,
            path_effects=[path_effects.withStroke(linewidth=2.4, foreground="#0d366b")],
        )

    sm = plt.cm.ScalarMappable(cmap=BLUE_RAMP, norm=Normalize(0, vmax))
    cb = ax.figure.colorbar(sm, ax=ax, orientation="horizontal",
                            fraction=0.04, pad=0.02, shrink=0.72)
    cb.set_label("Transactions per MSOA", color=INK_2, fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)


def draw_rank(ax, gdf) -> None:
    top = gdf.sort_values("count").tail(12)
    colors = BLUE_RAMP(0.25 + 0.7 * (top["count"] / top["count"].max()))
    ax.barh(top["short"], top["count"], color=colors, height=0.74)
    for y, (c, p) in enumerate(zip(top["count"], top["median_price"])):
        ax.text(c - top["count"].max() * 0.015, y, f"{int(c):,}", va="center", ha="right",
                fontsize=8.2, color="white", fontweight="bold")
        ax.text(top["count"].max() * 0.02, y, f"£{p/1000:.0f}k median", va="center",
                ha="left", fontsize=7.2, color="#dbe9fb")
    style_axes(ax)
    ax.set_xlim(0, top["count"].max() * 1.02)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8.5, colors=INK_2)
    ax.set_title("Busiest MSOAs by transaction count", fontsize=11.5, fontweight="bold",
                 color=INK, loc="left", pad=24)
    ax.text(0, 1.008, "ranked by number of sales, labelled with the area median price",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def draw_drivers(ax, df) -> None:
    g = df.groupby("msoa21")
    cnt = g.size()
    demo = g.first(numeric_only=True)
    demo["median_price"] = g["price_sold"].median()
    rows = []
    for col, label in TRAITS:
        r = np.corrcoef(cnt.values, demo.loc[cnt.index, col].values)[0, 1]
        rows.append((label, r))
    rows.sort(key=lambda t: t[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [POS if v >= 0 else NEG for v in vals]
    ax.barh(labels, vals, color=colors, height=0.72)
    for y, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.8, color=INK_2)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    style_axes(ax)
    ax.set_xlim(-0.62, 0.62)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8.5, colors=INK_2)
    ax.set_title("What marks a high-volume MSOA", fontsize=11.5, fontweight="bold",
                 color=INK, loc="left", pad=24)
    ax.text(0, 1.008, "area trait vs sales count  ·  blue = busier, red = quieter",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def draw_mix(ax, df, gdf) -> None:
    order = gdf.sort_values("count", ascending=False)["MSOA21CD"].head(12).tolist()
    rows = ["All Cambridgeshire"] + gdf.set_index("MSOA21CD").loc[order, "short"].tolist()

    def comp(sub):
        v = sub["grp"].value_counts(normalize=True) * 100
        return [v.get(g, 0.0) for g in PROP_ORDER]

    data = [comp(df)] + [comp(df[df["msoa21"] == m]) for m in order]
    data = np.array(data)
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for j, g in enumerate(PROP_ORDER):
        ax.barh(y, data[:, j], left=left, color=PROP_COLOR[g], height=0.72,
                edgecolor=SURFACE, linewidth=1.2)
        left += data[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=8.3, color=INK_2)
    # emphasise the reference row
    ax.get_yticklabels()[list(y).index(len(rows) - 1)].set_fontweight("bold")
    ax.get_yticklabels()[list(y).index(len(rows) - 1)].set_color(INK)
    style_axes(ax)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8, color=MUTED)
    ax.set_title("Property-type allocation of the busiest MSOAs", fontsize=11.5,
                 fontweight="bold", color=INK, loc="left", pad=24)
    ax.text(0, 1.02, "share of sales by dwelling type, top 12 MSOAs vs the county overall",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    handles = [Patch(facecolor=PROP_COLOR[g], label=g) for g in PROP_ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=6, frameon=False, fontsize=8.5, handlelength=1.1,
              columnspacing=1.3, labelcolor=INK_2)


def base_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def build_dashboard(df, gdf) -> Path:
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.05, 1.05],
                          height_ratios=[1.18, 1.0], hspace=0.30, wspace=0.27,
                          left=0.035, right=0.975, top=0.865, bottom=0.075)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_drv = fig.add_subplot(gs[0, 2])
    ax_mix = fig.add_subplot(gs[1, 1:])

    draw_map(ax_map, gdf)
    ax_map.set_title("Where the sales are — transactions per MSOA", fontsize=11.5,
                     fontweight="bold", color=INK, loc="left", pad=8)
    draw_rank(ax_rank, gdf)
    draw_drivers(ax_drv, df)
    draw_mix(ax_mix, df, gdf)

    fig.text(0.035, 0.955, "Cambridgeshire Property Sales", fontsize=25,
             fontweight="bold", color=INK)
    fig.text(0.035, 0.915,
             f"{len(df):,} sales across {gdf.shape[0]} MSOAs (2018-2022)  ·  "
             "geography, the traits behind volume, and the housing mix",
             fontsize=12.5, color=INK_2)
    fig.text(0.035, 0.028,
             "Source: cleaned Cambridgeshire property sales  ·  boundaries: ONS "
             "Middle-layer Super Output Areas (Dec 2021)  ·  British National Grid (EPSG:27700)",
             fontsize=8.5, color=MUTED)

    out = IMAGE_DIR / "cambridgeshire_msoa_dashboard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def build_map(df, gdf) -> Path:
    fig = plt.figure(figsize=(12, 12.8))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.83])
    draw_map(ax, gdf, standalone=True)
    fig.text(0.03, 0.955, "Cambridgeshire Property Transactions by MSOA",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(0.03, 0.918,
             f"{len(df):,} recorded sales (2018-2022) across {gdf.shape[0]} areas  ·  "
             "darker = more transactions", fontsize=11.5, color=INK_2)
    fig.text(0.03, 0.02, "Boundaries: ONS MSOA 2021  ·  EPSG:27700", fontsize=8.5, color=MUTED)
    out = IMAGE_DIR / "cambridgeshire_transaction_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    base_style()
    df, gdf = load()
    p1 = build_dashboard(df, gdf)
    p2 = build_map(df, gdf)
    print(f"Saved {p1.relative_to(PROJECT_ROOT)}")
    print(f"Saved {p2.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
