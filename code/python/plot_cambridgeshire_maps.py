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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "image"

CLEANED = DATA_DIR / "cleaned" / "Cambridge_data_cleaned.csv"
POPULATION = DATA_DIR / "spatial" / "msoa_population_mid2022.csv"
NAMES = DATA_DIR / "spatial" / "msoa_names.csv"

# the cleaned file now uses R-style column names + distance columns; map back to the
# names used throughout this script.
COL_RENAME = {
    "Minors.0.18": "Minors 0-18", "Adults.18.60": "Adults 18-60", "Elders..60": "Elders >60",
    "X0.5km": "0-5km", "X5.30km": "5-30km", "X.30km": ">30km",
    "Work.from.Home..0km.": "Work from Home (0km)", "Living.Offshore": "Living Offshore",
    "Other.Qualification": "Other Qualification", "Work.from.Home": "Work from Home",
    "City.Public": "City Public", "Other.Method": "Other Method",
}
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
    df = pd.read_csv(CLEANED).rename(columns=COL_RENAME)
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


# --- 6. per-type 4-year trends: price (line) + volume (bars) ---------------------
YEARS = [2018, 2019, 2020, 2021, 2022]


def fig_type_trends(df) -> Path:
    d = df.copy()
    d["year"] = pd.to_datetime(d["sold_date"]).dt.year
    d = d[d["year"].isin(YEARS)]
    tot = d.groupby("property_type").size().sort_values(ascending=False)
    keep = [t for t in tot.index if tot[t] >= 100]  # drop the 7 sparse types (zero-sale years)

    ncol, blue, barc = 4, "#2a78d6", "#dfe7d9"
    nrow = -(-len(keep) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.9 * nrow))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.075, hspace=0.55, wspace=0.22)

    for i, t in enumerate(keep):
        ax = axes.flat[i]
        sub = d[d["property_type"] == t]
        vol = [int((sub["year"] == y).sum()) for y in YEARS]
        price = [sub.loc[sub["year"] == y, "price_sold"].median() / 1000 for y in YEARS]

        axb = ax.twinx()                      # volume context bars (kept subordinate)
        axb.bar(YEARS, vol, width=0.62, color=barc, zorder=1)
        axb.set_ylim(0, max(vol) * 3.2); axb.axis("off")

        ax.plot(YEARS, price, color=blue, lw=2.2, marker="o", ms=4.5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(YEARS[0], price[0], f"£{price[0]:.0f}k", va="bottom", ha="left",
                fontsize=8, color=INK_2, zorder=4)
        ax.text(YEARS[-1], price[-1], f"£{price[-1]:.0f}k", va="bottom", ha="right",
                fontsize=8, fontweight="bold", color=INK, zorder=4)
        ax.set_title(f"{t}  ·  n={int(tot[t]):,}", fontsize=9.5, fontweight="bold",
                     color=INK, loc="left", pad=6)
        ax.set_zorder(axb.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_xticks(YEARS)
        ax.set_xticklabels(["'18", "'19", "'20", "'21", "'22"], fontsize=8, color=MUTED)
        ax.set_xlim(2017.6, 2022.4)
        ax.set_ylim(min(price) * 0.85, max(price) * 1.18)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)

    for j in range(len(keep), nrow * ncol):
        axes.flat[j].axis("off")

    title_block(fig, "How Each Property Type Changed, 2018-2022",
                "median sale price (line, £000s) and yearly sales volume (bars) per type  ·  17 types with enough data")
    footer(fig, "Yearly totals partly reflect data coverage (2019 over-represented, 2022 to March only)  ·  7 rare types with zero-sale years omitted")
    out = IMAGE_DIR / "cambridgeshire_type_trends.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# ================= EDA / modelling-prep charts (each its own image) =============
DIVERGE = LinearSegmentedColormap.from_list("cam_div", ["#2a78d6", "#eef0ee", "#e34948"])


def gbp():
    return FuncFormatter(lambda v, _: f"£{v/1000:.0f}k")


def _bare(ax, keep_bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


HEATMAP_COLS = ["price_sold", "num_bed_", "num_bath", "num_reception", "Asian", "Black",
                "Mixed", "White", "Other", "Minors 0-18", "Adults 18-60", "Elders >60",
                "0-5km", "5-30km", ">30km", "Work from Home (0km)", "Living Offshore",
                "Low", "Medium", "High", "Other Qualification", "Work from Home",
                "City Public", "Rail", "Cycle", "Driving", "Foot", "Other Method"]
HEATMAP_LABELS = {
    "price_sold": "Price", "num_bed_": "Beds", "num_bath": "Baths", "num_reception": "Receptions",
    "Minors 0-18": "Age 0-18", "Adults 18-60": "Age 18-60", "Elders >60": "Age 60+",
    "0-5km": "Commute <5km", "5-30km": "Commute 5-30km", ">30km": "Commute >30km",
    "Work from Home (0km)": "Dist: WFH", "Living Offshore": "Offshore",
    "Low": "Qual: low", "Medium": "Qual: med", "High": "Qual: high",
    "Other Qualification": "Qual: other", "Work from Home": "Travel: WFH",
    "City Public": "Bus", "Rail": "Rail", "Cycle": "Cycle", "Driving": "Car",
    "Foot": "Foot", "Other Method": "Other travel"}


def fig_price_distribution(df) -> Path:
    p = df["price_sold"].values
    med = float(np.median(p))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.subplots_adjust(left=0.05, right=0.975, top=0.80, bottom=0.13, wspace=0.14)
    for ax, logx in zip(axes, (False, True)):
        bins = (np.logspace(np.log10(p.min()), np.log10(p.max()), 44) if logx
                else np.linspace(p.min(), p.max(), 46))
        ax.hist(p, bins=bins, color="#1baf7a" if logx else "#2a78d6",
                edgecolor="white", linewidth=0.4)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_locator(FixedLocator([50_000, 100_000, 200_000, 400_000, 700_000]))
        ax.axvline(med, color=NEG, lw=2)
        ax.xaxis.set_major_formatter(gbp())
        ax.set_title("Linear axis — right-skewed" if not logx else "Log axis — near-symmetric",
                     fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
        _bare(ax); ax.set_yticks([])
    axes[0].axvline(747500, color="#eb6834", lw=1.4, ls="--")
    axes[0].text(740000, axes[0].get_ylim()[1] * 0.9, "£747.5k max ", color="#eb6834",
                 fontsize=9, ha="right", va="top")
    axes[0].text(med - 12000, axes[0].get_ylim()[1] * 0.99, f"median £{med/1000:.0f}k ",
                 color=NEG, fontsize=9, ha="right", va="top", fontweight="bold")
    title_block(fig, "The Target: House-Price Distribution",
                "raw prices are right-skewed and bounded at £747,500 (the upper filter); on a log axis they are near-symmetric — model log(price)")
    footer(fig, f"{len(p):,} sales  ·  median £{med/1000:.0f}k  ·  range £{p.min()/1000:.0f}k–£{p.max()/1000:.0f}k (the max is a filter, not a pile-up)")
    out = IMAGE_DIR / "cambridgeshire_price_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_price_by_bedrooms(df) -> Path:
    beds = list(range(1, 8))
    data = [df.loc[df["num_bed_"] == b, "price_sold"].values for b in beds]
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.10)
    ax.boxplot(data, positions=beds, widths=0.62, patch_artist=True, showfliers=False,
               medianprops=dict(color="white", linewidth=2),
               boxprops=dict(facecolor="#2a78d6", edgecolor="#256abf"),
               whiskerprops=dict(color="#256abf"), capprops=dict(color="#256abf"))
    for b, d in zip(beds, data):
        ax.text(b, df["price_sold"].max() * 0.03, f"n={len(d):,}", ha="center",
                fontsize=8, color=MUTED)
    ax.yaxis.set_major_formatter(gbp())
    ax.set_xlabel("Number of bedrooms", fontsize=10.5, color=INK_2)
    _bare(ax); ax.tick_params(axis="y", labelsize=9)
    title_block(fig, "Price Rises with Bedrooms — then Hits the Cap",
                "sale price by bedroom count  ·  box = middle 50%, line = median, whiskers = typical range")
    footer(fig, "Outliers hidden for clarity  ·  the £747.5k ceiling flattens the top of the 5-7 bed boxes")
    out = IMAGE_DIR / "cambridgeshire_price_by_bedrooms.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_correlation_heatmap(df) -> Path:
    cols = [c for c in HEATMAP_COLS if c in df.columns]
    corr = df[cols].corr().values
    labels = [HEATMAP_LABELS.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(13, 12))
    fig.subplots_adjust(left=0.17, right=0.99, top=0.86, bottom=0.17)
    im = ax.imshow(corr, cmap=DIVERGE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7.6, color=INK_2)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7.6, color=INK_2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(labels), 1, fill=False, edgecolor=INK, lw=1.4))
    cb = fig.colorbar(im, fraction=0.036, pad=0.02)
    cb.set_label("Correlation (Pearson r)", color=INK_2, fontsize=10)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, length=0, labelsize=9)
    title_block(fig, "How the Features Relate — Correlation Matrix",
                "blue = move together, red = move apart. The top row is price; census blocks each sum to 100 → collinear")
    footer(fig, "Pearson r across all sales  ·  strong within-block correlation (age, commute, qualifications) signals redundant features")
    out = IMAGE_DIR / "cambridgeshire_correlation_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_price_by_type_box(df) -> Path:
    tot = df.groupby("property_type").size()
    keep = tot[tot >= 100].index
    med = (df[df["property_type"].isin(keep)].groupby("property_type")["price_sold"]
           .median().sort_values())
    types = med.index.tolist()
    data = [df.loc[df["property_type"] == t, "price_sold"].values for t in types]
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.87, bottom=0.07)
    ax.boxplot(data, orientation="horizontal", positions=range(len(types)), widths=0.62,
               patch_artist=True, showfliers=False,
               medianprops=dict(color="white", linewidth=2),
               boxprops=dict(facecolor="#1baf7a", edgecolor="#158f64"),
               whiskerprops=dict(color="#158f64"), capprops=dict(color="#158f64"))
    ax.set_yticks(range(len(types))); ax.set_yticklabels(types, fontsize=9.2, color=INK)
    ax.xaxis.set_major_formatter(gbp())
    _bare(ax); ax.tick_params(axis="y", length=0)
    title_block(fig, "Price Spread by Property Type",
                "sale-price distribution per type (17 with enough data), sorted by median  ·  boxes overlap a lot")
    footer(fig, "Box = middle 50%, line = median, whiskers = typical range; outliers hidden  ·  rare types (<100 sales) omitted")
    out = IMAGE_DIR / "cambridgeshire_price_by_type_box.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_price_per_bed_map(df, gdf) -> Path:
    ppb = df.assign(v=df["price_sold"] / df["num_bed_"]).groupby("msoa21")["v"].median()
    g = gdf.assign(ppb=gdf["MSOA21CD"].map(ppb))
    norm = Normalize(g["ppb"].min(), g["ppb"].max())
    fig, ax = _choropleth(g, "ppb", TEAL_RAMP, norm)
    _hbar(fig, TEAL_RAMP, norm, "Median price per bedroom", lambda v, _: f"£{v/1000:.0f}k")
    title_block(fig, "Location Premium — Price per Bedroom",
                "median sale price divided by bedrooms  ·  strips out house size to show the pure location value")
    footer(fig, "Price per bedroom = median of (sale price / bedrooms) per area  ·  ONS MSOA 2021  ·  EPSG:27700")
    out = IMAGE_DIR / "cambridgeshire_price_per_bedroom_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out


def fig_sales_timeline(df) -> Path:
    d = pd.to_datetime(df["sold_date"])
    m = d.dt.to_period("M").value_counts().sort_index()
    x = np.arange(len(m))
    is_mar = [p.month == 3 for p in m.index]
    colors = ["#eb6834" if mar else "#9ec5f4" for mar in is_mar]
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.12)
    ax.bar(x, m.values, color=colors, width=0.9)
    # year ticks
    jans = [i for i, p in enumerate(m.index) if p.month == 1]
    ax.set_xticks(jans); ax.set_xticklabels([str(m.index[i].year) for i in jans], fontsize=10, color=INK_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    _bare(ax)
    for i, p in enumerate(m.index):
        if p.month == 3 and m.values[i] > 1500:
            ax.text(x[i], m.values[i] + 80, f"31 Mar\n{m.values[i]:,}", ha="center",
                    fontsize=8.5, color="#c2410c", fontweight="bold")
    ax.scatter([], [], marker="s", s=90, color="#eb6834", label="March (financial year-end)")
    ax.scatter([], [], marker="s", s=90, color="#9ec5f4", label="other months")
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK_2)
    title_block(fig, "When Sales Were Recorded — the 31-March Artifact",
                "monthly sales counts  ·  huge spikes on 31 March 2019 & 2020 are imputed year-end dates, not real activity")
    footer(fig, "By sold_date  ·  2022 runs only to March  ·  treat dates as coarse (financial year) not exact days when modelling")
    out = IMAGE_DIR / "cambridgeshire_sales_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


SOCIAL_VARS = ["Asian", "Black", "Mixed", "White", "Other", "Minors 0-18", "Adults 18-60",
               "Elders >60", "0-5km", "5-30km", ">30km", "Work from Home (0km)",
               "Living Offshore", "Low", "Medium", "High", "Other Qualification",
               "Work from Home", "City Public", "Rail", "Cycle", "Driving", "Foot", "Other Method"]
SOCIAL_LABEL = {
    "Work from Home (0km)": "Work from home (commute band)", "Work from Home": "Work from home (travel)",
    "Low": "Low qualifications", "High": "Degree-level qualifications",
    "Living Offshore": "Offshore workers", "Driving": "Drives to work",
    "Other Qualification": "Other qualifications", ">30km": "Commutes >30 km",
    "5-30km": "Commutes 5-30 km"}


def fig_social_vs_location(df) -> Path:
    """Show that the strong social→price correlations are absorbed by the two
    location distances — justifying a property+location model with no social features."""
    price = df["price_sold"].to_numpy(float)
    L = np.column_stack([np.ones(len(df)), df["dist_london"].to_numpy(float),
                         df["dist_cambridge"].to_numpy(float)])

    def resid(y):
        beta, *_ = np.linalg.lstsq(L, y, rcond=None)
        return y - L @ beta

    rp = resid(price)
    rows = []
    for s in SOCIAL_VARS:
        x = df[s].to_numpy(float)
        raw = np.corrcoef(x, price)[0, 1]
        part = np.corrcoef(resid(x), rp)[0, 1]
        rows.append((s, abs(raw), abs(part)))
    strong = sorted([r for r in rows if r[1] >= 0.15], key=lambda r: r[1])
    labels = [SOCIAL_LABEL.get(s, s) for s, _, _ in strong]
    raw_abs = [r[1] for r in strong]
    part_abs = [r[2] for r in strong]
    y = np.arange(len(strong))
    avg_expl = np.mean([(1 - p / r) * 100 for _, r, p in strong])

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    fig.subplots_adjust(left=0.28, right=0.955, top=0.80, bottom=0.11)
    for i in y:
        ax.plot([part_abs[i], raw_abs[i]], [i, i], color="#dfe3ea", lw=3.5, zorder=1,
                solid_capstyle="round")
    ax.scatter(part_abs, y, s=150, color="#c9c7bf", zorder=3, edgecolor="white", linewidth=1)
    ax.scatter(raw_abs, y, s=160, color="#2a78d6", zorder=3, edgecolor="white", linewidth=1)
    for i in y:
        ax.text(raw_abs[i] + 0.006, i, f"{raw_abs[i]:.2f}", va="center", ha="left",
                fontsize=8.8, color="#2a78d6", fontweight="bold")
        ax.text(part_abs[i] - 0.006, i, f"{part_abs[i]:.2f}", va="center", ha="right",
                fontsize=8.8, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(strong) - 0.4)
    ax.set_xlim(0, max(raw_abs) * 1.22)
    _bare(ax); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Correlation with sale price (strength, |r|)", fontsize=10, color=INK_2)

    ax.scatter([], [], s=150, color="#2a78d6", label="correlation with price on its own")
    ax.scatter([], [], s=150, color="#c9c7bf", label="after removing distance to Cambridge & London")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=10, labelcolor=INK_2, columnspacing=2.2, handletextpad=0.5)

    title_block(fig, "Location Already Carries the Social Signal",
                f"each strong social trait's link to price collapses ~{avg_expl:.0f}% once the two distances are known — so a property + location model keeps the signal without any social variables")
    footer(fig, "Blue = raw |correlation| with price; grey = partial |correlation| after controlling for dist_london & dist_cambridge  ·  the gap is what location already explains")
    out = IMAGE_DIR / "cambridgeshire_social_vs_location.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_price_correlation_rank(df) -> Path:
    price = df["price_sold"].to_numpy(float)
    feats = [("num_bed_", "Bedrooms", "Property"), ("num_bath", "Bathrooms", "Property"),
             ("num_reception", "Receptions", "Property"),
             ("dist_cambridge", "Distance to Cambridge", "Location"),
             ("dist_london", "Distance to London", "Location")]
    feats += [(s, SOCIAL_LABEL.get(s, s), "Social") for s in SOCIAL_VARS]
    rows = [(label, cat, np.corrcoef(df[col].to_numpy(float), price)[0, 1])
            for col, label, cat in feats]
    rows.sort(key=lambda t: abs(t[2]))
    labels = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    catcol = {"Property": "#2a78d6", "Location": "#1baf7a", "Social": "#c9c7bf"}
    colors = [catcol[r[1]] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(12, 11))
    fig.subplots_adjust(left=0.30, right=0.93, top=0.855, bottom=0.06)
    ax.barh(y, [abs(v) for v in vals], color=colors, height=0.76)
    for i, v in enumerate(vals):
        ax.text(abs(v) + 0.004, i, f"{v:+.2f}", va="center", ha="left", fontsize=8.8, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.4, color=INK)
    ax.set_xlim(0, max(abs(v) for v in vals) * 1.16)
    _bare(ax); ax.set_xlabel("Correlation with sale price  (|r|; sign on the label)", fontsize=10, color=INK_2)
    handles = [Patch(color=catcol[c], label=c) for c in ("Property", "Location", "Social")]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10.5,
                    labelcolor=INK_2, title="Feature type", title_fontsize=10)
    leg.get_title().set_color(INK)
    title_block(fig, "What Correlates Most with Price",
                "every feature ranked by its correlation with sale price  ·  the model uses Property + Location only (Social shown for context)")
    footer(fig, "Pearson r across all sales  ·  bedrooms lead; the two distances rank near the top; excluded social traits are proxied by location")
    out = IMAGE_DIR / "cambridgeshire_price_correlation_rank.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_area_type_consistency(df) -> Path:
    names = pd.read_csv(NAMES).set_index("msoa21")["name"]
    groups = ["Detached", "Semi-detached", "Terraced", "Flat/Apartment", "Bungalow"]
    d = df[df["grp"].isin(groups)]
    agg = d.groupby(["msoa21", "grp"])["price_sold"].agg(["median", "size"]).reset_index()
    agg = agg[agg["size"] >= 5]
    mat = agg.pivot(index="msoa21", columns="grp", values="median").reindex(columns=groups)
    pct = mat.rank(pct=True) * 100
    order = pct.mean(axis=1).sort_values(ascending=False).index
    mat, pct = mat.loc[order], pct.loc[order]
    rowlabels = [names.get(c, c) for c in order]

    corr = mat.rank().corr(method="spearman").values
    iu = np.triu_indices(len(groups), 1)
    avg_r = np.nanmean(corr[iu])

    cmap = TEAL_RAMP.copy(); cmap.set_bad("#ececec")
    fig, ax = plt.subplots(figsize=(9.5, 14))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.05)
    im = ax.imshow(np.ma.masked_invalid(pct.values), cmap=cmap, aspect="auto", vmin=0, vmax=100)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v/1000:.0f}", ha="center", va="center", fontsize=6.6,
                        color="white" if pct.values[i, j] >= 55 else INK_2)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=9.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order))); ax.set_yticklabels(rowlabels, fontsize=7.4, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Area's price rank within each type (percentile)", color=INK_2, fontsize=9.5)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    title_block(fig, "Is an Expensive Area Expensive for Every Property Type?",
                f"median price (£000s) by area & type; shading = the area's rank within that type. Rows stay one colour → mostly yes (avg rank r = {avg_r:.2f})")
    footer(fig, "Areas sorted most→least expensive (top→bottom)  ·  cells need ≥5 sales, grey = too few  ·  ONS MSOA 2021")
    out = IMAGE_DIR / "cambridgeshire_area_type_consistency.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
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
    figs = [
        fig_count_map(df, gdf), fig_type_map(df, gdf), fig_price_profile(df),
        fig_price_map(df, gdf), fig_population_map(df, gdf), fig_type_trends(df),
        fig_price_distribution(df), fig_price_by_bedrooms(df), fig_correlation_heatmap(df),
        fig_price_by_type_box(df), fig_price_per_bed_map(df, gdf), fig_sales_timeline(df),
        fig_social_vs_location(df), fig_price_correlation_rank(df), fig_area_type_consistency(df),
    ]
    for p in figs:
        print(f"Saved {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
