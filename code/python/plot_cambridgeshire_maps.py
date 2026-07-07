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
POPULATION = DATA_DIR / "spatial" / "msoa_population_mid2022.csv"
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

# traits compared between the priciest and most affordable areas (all percentages)
PRICE_TRAITS = [
    ("High", "Adults with a degree"),
    ("Work from Home", "Work from home"),
    ("0-5km", "Live within 5km of work"),
    ("Cycle", "Cycle to work"),
    ("Foot", "Walk to work"),
    ("Driving", "Drive to work"),
]
GROUP_AFFORD = "#eb6834"  # orange - most affordable areas
GROUP_PRICEY = "#2a78d6"  # blue   - priciest areas

VIOLET_RAMP = LinearSegmentedColormap.from_list(
    "cam_violet",
    ["#f0ecfb", "#ddd0f3", "#c2ade8", "#a385db", "#845fcc", "#653fb0", "#4c2c8f", "#35206b"],
)

# Real towns to label on every map: (name, lat, lon, dx, dy, ha) — dx/dy are the
# label offset in points from the dot, hand-tuned so labels sit clear of one another.
TOWNS = [
    ("Cambridge", 52.2053, 0.1218, 16, 10, "left"),
    ("Ely", 52.3993, 0.2626, 10, 8, "left"),
    ("Soham", 52.3330, 0.3363, 10, 6, "left"),
    ("Littleport", 52.4568, 0.3046, 10, 8, "left"),
    ("Cambourne", 52.2200, -0.0730, -10, 6, "right"),
    ("Sawston", 52.1210, 0.1690, 9, -9, "left"),
    ("Waterbeach", 52.2670, 0.1920, 9, 7, "left"),
    ("Cottenham", 52.2880, 0.1270, 0, 11, "center"),
    ("Burwell", 52.2750, 0.3280, 9, 6, "left"),
    ("Bar Hill", 52.2480, -0.0430, -9, 7, "right"),
    ("Melbourn", 52.0820, 0.0220, 0, -12, "center"),
    ("Linton", 52.0980, 0.2840, 9, -6, "left"),
    ("Fulbourn", 52.1780, 0.2220, 9, -8, "left"),
    # the priciest area (£580k median) — labelled so its dark polygon is identified
    ("Hardwick & Highfields", 52.1852, -0.0128, 0, -12, "center"),
]

# The two headline landmarks — starred and made prominent ("where we are now").
SPECIAL = [
    ("Cambridge University", 52.2043, 0.1160, 4, -46, "center"),
    ("Isaac Newton Institute", 52.2109, 0.0985, -50, 26, "right"),
]
STAR = "#d4380d"  # strong orange-red for the two headline landmarks


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


def _project(items):
    """Project a list of (name, lat, lon, ...) to EPSG:27700 points."""
    lon = [t[2] for t in items]
    lat = [t[1] for t in items]
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(27700)


def annotate_places(ax) -> None:
    """Label the main towns (dots) plus the two headline landmarks (stars)."""
    halo = [path_effects.withStroke(linewidth=3, foreground=SURFACE)]
    for (name, la, lo, dx, dy, ha), geom in zip(TOWNS, _project(TOWNS).geometry):
        ax.plot(geom.x, geom.y, "o", ms=5, mfc=INK, mec="white", mew=0.8, zorder=6)
        ax.annotate(name, (geom.x, geom.y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=9.5, fontweight="bold", color=INK,
                    zorder=7, path_effects=halo)
    for (name, la, lo, dx, dy, ha), geom in zip(SPECIAL, _project(SPECIAL).geometry):
        ax.plot(geom.x, geom.y, marker="*", ms=22, mfc=STAR, mec="white", mew=1.4, zorder=8)
        ax.annotate(name, (geom.x, geom.y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=10.5, fontweight="bold", color=STAR, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=STAR, lw=1.1, shrinkA=0, shrinkB=6),
                    path_effects=halo)


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

    pop = pd.read_csv(POPULATION).set_index("msoa21")["population_2022"]

    gdf = gpd.read_file(BOUNDARIES, columns=["MSOA21CD", "MSOA21NM", "geometry"])
    gdf = gdf[gdf["MSOA21CD"].isin(per.index)].copy()
    gdf = gdf.merge(per, left_on="MSOA21CD", right_index=True).to_crs(epsg=27700)
    gdf["population"] = gdf["MSOA21CD"].map(pop)
    gdf["density"] = gdf["population"] / (gdf.geometry.area / 1e6)  # residents per km^2
    return df, gdf


# --- 1. transaction count map ---------------------------------------------------
def fig_count_map(df, gdf) -> Path:
    norm = Normalize(0, gdf["count"].max())
    fig, ax = _choropleth(gdf, "count", BLUE_RAMP, norm)
    _hbar(fig, BLUE_RAMP, norm, "Transactions per area", lambda v, _: f"{v:,.0f}")
    title_block(fig, "Where the Sales Are — Transactions by Area",
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
    annotate_places(ax)

    handles = [Patch(facecolor=TYPE_COLOR[t], label=t) for t in TYPE_ORDER]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=11,
              title="Most common dwelling type", title_fontsize=11,
              labelcolor=INK_2, handlelength=1.2)
    ax.get_legend().get_title().set_color(INK)

    title_block(fig, "Housing Character — Dominant Dwelling Type by Area",
                "the most-sold property type in each area  ·  flats cluster in the city, detached homes in the countryside")
    footer(fig, "Types grouped from 24 raw categories  ·  Boundaries: ONS MSOA 2021  ·  EPSG:27700")
    out = IMAGE_DIR / "cambridgeshire_property_type_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


# --- 3. what makes an area expensive (priciest vs affordable profile) ------------
def fig_price_profile(df) -> Path:
    # split the 43 MSOAs into price thirds and average each trait per group
    demo = [c for c, _ in PRICE_TRAITS]
    per = df.groupby("msoa21").agg(price=("price_sold", "median"),
                                   **{c: (c, "first") for c in demo}).sort_values("price")
    k = len(per) // 3
    afford, pricey = per.iloc[:k], per.iloc[-k:]
    rows = [(label, afford[col].mean(), pricey[col].mean()) for col, label in PRICE_TRAITS]
    rows.sort(key=lambda r: r[2] - r[1])  # biggest "more in cheap areas" at bottom
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))

    fig = plt.figure(figsize=(11.5, 7.6))
    ax = fig.add_axes([0.28, 0.12, 0.66, 0.66])
    for i, (_, a, p) in enumerate(rows):
        ax.plot([a, p], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter([a], [i], s=190, color=GROUP_AFFORD, zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter([p], [i], s=190, color=GROUP_PRICEY, zorder=3, edgecolor="white", linewidth=1.5)
        lo, hi = (a, p) if a <= p else (p, a)
        lo_c, hi_c = (GROUP_AFFORD, GROUP_PRICEY) if a <= p else (GROUP_PRICEY, GROUP_AFFORD)
        ax.text(lo - 1.1, i, f"{lo:.0f}%", va="center", ha="right", fontsize=10.5,
                color=lo_c, fontweight="bold")
        ax.text(hi + 1.1, i, f"{hi:.0f}%", va="center", ha="left", fontsize=10.5,
                color=hi_c, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11.5, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, max(p for _, _, p in rows) * 1.35)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # legend as two labelled swatches
    ax.scatter([], [], s=170, color=GROUP_AFFORD, label="Most affordable areas  (~£318k)")
    ax.scatter([], [], s=170, color=GROUP_PRICEY, label="Priciest areas  (~£447k)")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=INK_2,
              handletextpad=0.4, borderaxespad=0.3)

    title_block(fig, "What Makes an Area Expensive?",
                "average neighbourhood profile of Cambridgeshire's 14 priciest vs 14 most affordable areas")
    footer(fig, "Each area = one MSOA (of 43), split into price thirds  ·  figures are area-level census & commute shares")
    out = IMAGE_DIR / "cambridgeshire_price_drivers.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


def _choropleth(gdf, column, cmap, norm):
    """Shared full-bleed map canvas with town/landmark labels; returns (fig, ax)."""
    fig = plt.figure(figsize=(12, 12.8))
    ax = fig.add_axes([0.02, 0.10, 0.96, 0.78])
    gdf.plot(ax=ax, column=column, cmap=cmap, norm=norm, linewidth=0.6, edgecolor="white")
    ax.set_axis_off(); ax.set_aspect("equal")
    annotate_places(ax)
    return fig, ax


def _hbar(fig, cmap, norm, label, fmt):
    """Horizontal colorbar in a fixed axes above the footer (no overlap)."""
    cax = fig.add_axes([0.28, 0.072, 0.44, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax,
                      orientation="horizontal")
    cb.set_label(label, color=INK_2, fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)


# --- 4. median price map --------------------------------------------------------
def fig_price_map(df, gdf) -> Path:
    norm = Normalize(gdf["median_price"].min(), gdf["median_price"].max())
    fig, ax = _choropleth(gdf, "median_price", TEAL_RAMP, norm)
    _hbar(fig, TEAL_RAMP, norm, "Median sale price", lambda v, _: f"£{v/1000:.0f}k")
    title_block(fig, "House Prices across Cambridgeshire",
                "median sale price by area (2018-2022)  ·  priciest in Cambridge & the villages west of the city")
    footer(fig, "Price = median of recorded sales  ·  Boundaries: ONS MSOA 2021  ·  EPSG:27700")
    out = IMAGE_DIR / "cambridgeshire_price_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


# --- 5. population density map (official ONS mid-2022) --------------------------
def fig_population_map(df, gdf) -> Path:
    norm = Normalize(0, gdf["density"].max())
    fig, ax = _choropleth(gdf, "density", VIOLET_RAMP, norm)
    _hbar(fig, VIOLET_RAMP, norm, "Residents per km² (mid-2022)",
          lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    total = int(gdf["population"].sum())
    title_block(fig, "Where People Live — Population Density",
                f"{total:,} residents (ONS mid-2022) across {gdf.shape[0]} areas  ·  packed into Cambridge, sparse in the fens & countryside")
    footer(fig, "Population: ONS mid-2022 MSOA estimates (Nomis NM_2014_1)  ·  density = residents / land area  ·  ONS MSOA 2021  ·  EPSG:27700")
    out = IMAGE_DIR / "cambridgeshire_population_map.png"
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
    for p in (fig_count_map(df, gdf), fig_type_map(df, gdf), fig_price_profile(df),
              fig_price_map(df, gdf), fig_population_map(df, gdf)):
        print(f"Saved {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
