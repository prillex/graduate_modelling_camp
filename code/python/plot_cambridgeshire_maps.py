"""Cambridgeshire property-sales maps and charts.

Each figure is written as its own standalone image in ``image/``:

  1. cambridgeshire_transaction_count_map.png - choropleth of sales per MSOA.
  2. cambridgeshire_property_type_map.png     - dominant dwelling type per MSOA.
  3. cambridgeshire_volume_drivers.png        - area traits that correlate with
     transaction volume (the "what marks a high-volume MSOA" chart).
  4. cambridgeshire_region_landmarks_map.png  - median price per MSOA annotated
     with the landmarks and regional characteristics behind price and turnover.

Colour follows the job it does: single-hue ramps for magnitude (blue = counts,
teal = price), a fixed-order categorical set for dwelling types, and a
blue<->red diverging scale for the trait correlations.
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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

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
POS = "#2a78d6"  # blue - "more transactions"
NEG = "#e34948"  # red  - "fewer transactions"
ACCENT = "#eb6834"  # orange - landmark markers

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "cam_blue",
    ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
TEAL_RAMP = LinearSegmentedColormap.from_list(
    "cam_teal",
    ["#e4f6ef", "#bde9d8", "#8ddabd", "#54c6a0", "#1baf7a", "#158f64", "#0e6f4e", "#0a5038"],
)

# fixed-order categorical slots for dwelling types
TYPE_ORDER = ["Detached", "Semi-detached", "Terraced", "Flat/Apartment"]
TYPE_COLOR = {
    "Detached": "#2a78d6",
    "Semi-detached": "#1baf7a",
    "Terraced": "#eda100",
    "Flat/Apartment": "#4a3aa7",
}

# 24 raw property types -> display groups
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

# Key Cambridgeshire landmarks (num, lat, lon, name, one-line effect note). Shown
# as numbered dots on the map, described in the side key.
LANDMARKS = [
    (1, 52.2045, 0.1170, "University of Cambridge", "historic colleges: priciest, low turnover"),
    (2, 52.1745, 0.1400, "Biomedical Campus", "Addenbrooke's: major employer, south"),
    (3, 52.2320, 0.1490, "Cambridge Science Park", "'Silicon Fen' tech jobs, north"),
    (4, 52.2220, -0.0730, "Cambourne", "new town: new-build boom, top volume"),
    (5, 52.3990, 0.2620, "Ely", "cathedral commuter city (East Cambs)"),
    (6, 52.3330, 0.3360, "Soham", "affordable market town, high turnover"),
    (7, 52.4620, 0.3050, "Littleport", "cheapest, high-volume town (north)"),
    (8, 52.1850, -0.0050, "West villages", "Grantchester/Comberton: £580k, low turnover"),
]

REGION_NOTES = [
    ("Cambridge city  (14 MSOAs)", "high price · flat-heavy · low turnover"),
    ("South Cambs  (19 MSOAs)", "commuter villages + new towns (Cambourne)"),
    ("East Cambs  (10 MSOAs)", "affordable market towns: Ely, Soham, Littleport"),
]


def style_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def title_block(fig, title, subtitle) -> None:
    fig.text(0.03, 0.955, title, fontsize=19, fontweight="bold", color=INK)
    fig.text(0.03, 0.917, subtitle, fontsize=11.5, color=INK_2)


def footer(fig, text) -> None:
    fig.text(0.03, 0.02, text, fontsize=8.5, color=MUTED)


def label_top(ax, gdf, n, *, color="white", stroke="#0d366b") -> None:
    for _, row in gdf.sort_values("count", ascending=False).head(n).iterrows():
        pt = row.geometry.representative_point()
        ax.annotate(f"{row['short']}\n{int(row['count']):,}", (pt.x, pt.y),
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=color, linespacing=1.05,
                    path_effects=[path_effects.withStroke(linewidth=2.4, foreground=stroke)])


def load() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    df = pd.read_csv(CLEANED)
    df["grp"] = df["property_type"].map(PROP_GROUP)

    demo = ["Work from Home", "Cycle", "Foot", "0-5km", "High", "Low", "Minors 0-18", "Driving"]
    per = df.groupby("msoa21").agg(
        count=("price_sold", "size"),
        median_price=("price_sold", "median"),
        **{c: (c, "first") for c in demo},
    )
    per["dominant"] = df.groupby("msoa21")["grp"].agg(lambda s: s.value_counts().idxmax())

    gdf = gpd.read_file(BOUNDARIES, columns=["MSOA21CD", "MSOA21NM", "geometry"])
    gdf = gdf[gdf["MSOA21CD"].isin(per.index)].copy()
    gdf = gdf.merge(per, left_on="MSOA21CD", right_index=True).to_crs(epsg=27700)
    gdf["short"] = (gdf["MSOA21NM"]
                    .str.replace("South Cambridgeshire", "S. Cambs", regex=False)
                    .str.replace("East Cambridgeshire", "E. Cambs", regex=False))
    return df, gdf


# --- 1. transaction count map ---------------------------------------------------
def fig_count_map(df, gdf) -> Path:
    fig = plt.figure(figsize=(12, 12.8))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.83])
    vmax = gdf["count"].max()
    gdf.plot(ax=ax, column="count", cmap=BLUE_RAMP, norm=Normalize(0, vmax),
             linewidth=0.6, edgecolor="white")
    ax.set_axis_off(); ax.set_aspect("equal")
    label_top(ax, gdf, 8)

    sm = plt.cm.ScalarMappable(cmap=BLUE_RAMP, norm=Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.72)
    cb.set_label("Transactions per MSOA", color=INK_2, fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)

    title_block(fig, "Where the Sales Are — Transactions by MSOA",
                f"{len(df):,} recorded sales (2018-2022) across {gdf.shape[0]} areas  ·  darker = more transactions")
    footer(fig, "Boundaries: ONS MSOA 2021  ·  EPSG:27700")
    out = IMAGE_DIR / "cambridgeshire_transaction_count_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


# --- 2. dominant property-type map ----------------------------------------------
def fig_type_map(df, gdf) -> Path:
    fig = plt.figure(figsize=(12, 12.8))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.83])
    cmap = ListedColormap([TYPE_COLOR[t] for t in TYPE_ORDER])
    codes = gdf["dominant"].map({t: i for i, t in enumerate(TYPE_ORDER)})
    gdf.assign(_c=codes).plot(ax=ax, column="_c", cmap=cmap,
                              norm=BoundaryNorm(range(len(TYPE_ORDER) + 1), cmap.N),
                              linewidth=0.6, edgecolor="white")
    ax.set_axis_off(); ax.set_aspect("equal")

    handles = [Patch(facecolor=TYPE_COLOR[t], label=t) for t in TYPE_ORDER]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=11,
              title="Most common dwelling type", title_fontsize=11,
              labelcolor=INK_2, handlelength=1.2)
    ax.get_legend().get_title().set_color(INK)

    title_block(fig, "Housing Character — Dominant Dwelling Type by MSOA",
                "the most-sold property type in each area  ·  flats cluster in the city, detached homes in the countryside")
    footer(fig, "Types grouped from 24 raw categories  ·  Boundaries: ONS MSOA 2021  ·  EPSG:27700")
    out = IMAGE_DIR / "cambridgeshire_property_type_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


# --- 3. volume drivers chart ----------------------------------------------------
def fig_drivers(df) -> Path:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.30, 0.10, 0.63, 0.74])
    g = df.groupby("msoa21")
    cnt = g.size()
    demo = g.first(numeric_only=True)
    demo["median_price"] = g["price_sold"].median()
    rows = sorted(((label, np.corrcoef(cnt.values, demo.loc[cnt.index, col].values)[0, 1])
                   for col, label in TRAITS), key=lambda t: t[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [POS if v >= 0 else NEG for v in vals]
    ax.barh(labels, vals, color=colors, height=0.72)
    for y, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, color=INK_2)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style_axes(ax)
    ax.set_xlim(-0.62, 0.62); ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=11, colors=INK_2)

    title_block(fig, "What Marks a High-Volume MSOA",
                "correlation of an area's traits with its transaction count  ·  blue = busier, red = quieter")
    footer(fig, "Pearson r across 43 MSOAs  ·  traits are area-level census/commute measures")
    out = IMAGE_DIR / "cambridgeshire_volume_drivers.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


# --- 4. region + landmarks price map --------------------------------------------
def fig_landmarks_map(df, gdf) -> Path:
    fig = plt.figure(figsize=(16, 12.5))
    ax = fig.add_axes([0.36, 0.05, 0.60, 0.82])  # map on the right
    vmin, vmax = gdf["median_price"].min(), gdf["median_price"].max()
    gdf.plot(ax=ax, column="median_price", cmap=TEAL_RAMP, norm=Normalize(vmin, vmax),
             linewidth=0.6, edgecolor="white")
    ax.set_axis_off(); ax.set_aspect("equal")

    # numbered landmark dots, projected to the map CRS
    lon = [m[2] for m in LANDMARKS]
    lat = [m[1] for m in LANDMARKS]
    gp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(27700)
    for (num, *_), geom in zip(LANDMARKS, gp.geometry):
        ax.plot(geom.x, geom.y, "o", ms=15, mfc=ACCENT, mec="white", mew=1.8, zorder=5)
        ax.annotate(str(num), (geom.x, geom.y), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=6)

    sm = plt.cm.ScalarMappable(cmap=TEAL_RAMP, norm=Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.035, pad=0.01, shrink=0.5)
    cb.set_label("Median sale price", color=INK_2, fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"£{v/1000:.0f}k"))
    cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)

    # --- left panel: key + regional character (fig coords, clear of the map) ---
    fig.text(0.03, 0.86, "KEY LANDMARKS", fontsize=12, fontweight="bold", color=INK)
    y = 0.825
    for num, la, lo, name, note in LANDMARKS:
        fig.text(0.03, y, f"{num}", fontsize=10.5, fontweight="bold", color=ACCENT)
        fig.text(0.055, y, name, fontsize=10.5, fontweight="bold", color=INK_2)
        fig.text(0.055, y - 0.022, note, fontsize=8.8, color=MUTED)
        y -= 0.05

    fig.text(0.03, y - 0.01, "REGIONAL CHARACTER", fontsize=12, fontweight="bold", color=INK)
    y -= 0.05
    for head, note in REGION_NOTES:
        fig.text(0.03, y, head, fontsize=10, fontweight="bold", color=INK_2)
        fig.text(0.03, y - 0.022, note, fontsize=8.8, color=MUTED)
        y -= 0.052

    title_block(fig, "Why Prices & Sales Differ — Regions & Landmarks",
                "median sale price by MSOA, with the employers, towns and villages that shape demand")
    footer(fig, "Landmark positions approximate  ·  price = median of recorded sales 2018-2022  ·  ONS MSOA 2021  ·  EPSG:27700")
    out = IMAGE_DIR / "cambridgeshire_region_landmarks_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


def base_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    base_style()
    df, gdf = load()
    for p in (fig_count_map(df, gdf), fig_type_map(df, gdf), fig_drivers(df),
              fig_landmarks_map(df, gdf)):
        print(f"Saved {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
