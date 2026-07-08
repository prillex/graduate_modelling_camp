"""Two figures built on previously-run results.

1. image/rf_full_features_msoa_r2_map.png
   The per-MSOA test R2 of the Random Forest "full features + Cambridge/London
   distance" model, drawn on the Cambridgeshire map instead of a bar chart. R2
   values are read from the committed regional-results text file.

2. image/msoa_diversity_vs_price.png
   Neighbourhood diversity (Simpson / Gini-Simpson index) of each social block
   versus median house price across the 43 MSOAs, to show how little of the
   price signal the diversity measures actually carry.

Reuses the map canvas, palette and landmark labelling from
plot_cambridgeshire_maps so both figures match the rest of the image set.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cambridgeshire_maps import (  # noqa: E402
    IMAGE_DIR,
    PROJECT_ROOT,
    SURFACE,
    INK,
    INK_2,
    MUTED,
    GRID,
    POS,
    NEG,
    ACCENT,
    BLUE_RAMP,
    load,
    _choropleth,
    _hbar,
    title_block,
    footer,
)

R2_RESULT = (
    PROJECT_ROOT / "outputs" / "msoa_price_map" / "result of RF for different region.txt"
)
MODEL_NAME = "RF Full Features + Cambridge/London Distance"

# One Simpson index per social block. Column names match the readable names
# produced by plot_cambridgeshire_maps' COL_RENAME (applied inside load()).
SIMPSON_BLOCKS = {
    "Ethnicity": ["Asian", "Black", "Mixed", "White", "Other"],
    "Age": ["Minors 0-18", "Adults 18-60", "Elders >60"],
    "Education": ["Low", "Medium", "High", "Other Qualification"],
    "Commute distance": ["Work from Home (0km)", "0-5km", "5-30km", ">30km", "Living Offshore"],
    "Commute method": ["City Public", "Rail", "Cycle", "Driving", "Foot", "Other Method"],
}


def parse_regional_r2(path=R2_RESULT, model_name=MODEL_NAME):
    """Read per-MSOA R2 (price scale) for one model from the results text file."""
    rows = {}
    in_section = False
    header_seen = False
    for line in path.read_text().splitlines():
        if line.startswith("Regional performance by msoa21:"):
            in_section = model_name in line
            header_seen = False
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("msoa21"):
            header_seen = True
            continue
        if header_seen and re.match(r"^E0\d{7}\b", stripped):
            fields = stripped.split()
            rows[fields[0]] = float(fields[-1])  # last column = R2_price
    if not rows:
        raise ValueError(f"No rows parsed for model '{model_name}' in {path}")
    return pd.Series(rows, name="r2_price")


def simpson_index(block):
    """Gini-Simpson diversity, normalised to [0, 1], on row-summed proportions."""
    p = block.div(block.sum(axis=1), axis=0)
    k = p.shape[1]
    return (1 - (p ** 2).sum(axis=1)) / (1 - 1 / k)


def per_msoa_diversity(df):
    """One row per MSOA: median price plus the five Simpson diversity indices."""
    per = df.groupby("msoa21").first()
    per["median_price"] = df.groupby("msoa21")["price_sold"].median()
    per["n_sales"] = df.groupby("msoa21").size()
    for name, cols in SIMPSON_BLOCKS.items():
        per[name] = simpson_index(per[cols])
    return per


# --- figure 1: regional R2 map --------------------------------------------------
def fig_r2_map(gdf, r2):
    g = gdf.copy()
    g["r2_price"] = g["MSOA21CD"].map(r2)
    missing = g["r2_price"].isna().sum()
    if missing:
        raise ValueError(f"{missing} MSOAs on the map have no R2 value")

    norm = Normalize(g["r2_price"].min(), g["r2_price"].max())
    fig, ax = _choropleth(g, "r2_price", BLUE_RAMP, norm)
    _hbar(fig, BLUE_RAMP, norm, "Test R² (price scale)", lambda v, _: f"{v:.2f}")

    best = g.loc[g["r2_price"].idxmax()]
    worst = g.loc[g["r2_price"].idxmin()]
    title_block(
        fig,
        "How Well the Model Predicts, by Area",
        "Random Forest test R² per MSOA — full features + distance to Cambridge & London  ·  darker = better fit",
    )
    footer(
        fig,
        f"R² on a 20% hold-out set, per MSOA  ·  best {best['r2_price']:.2f}, "
        f"weakest {worst['r2_price']:.2f}  ·  Boundaries: ONS MSOA 2021  ·  EPSG:27700",
    )
    out = IMAGE_DIR / "rf_full_features_msoa_r2_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 2: diversity vs price ----------------------------------------------
def _scatter(ax, x, y, name, yfmt=lambda v, _: f"£{v/1000:.0f}k"):
    r = float(np.corrcoef(x, y)[0, 1])
    ax.scatter(x, y, s=34, color=POS, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color=INK, lw=1.4, ls="--", zorder=4)
    ax.set_title(name, fontsize=12.5, fontweight="bold", color=INK, pad=8, loc="left")
    ax.text(
        0.04,
        0.06,
        f"r = {r:+.2f}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=NEG if r < 0 else POS,
        ha="left",
        va="bottom",
    )
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.yaxis.set_major_formatter(yfmt)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7, zorder=0)
    return r


def _corr_summary_bar(ax, labels, vals, ref_label=None, ref_val=None, title="Correlation"):
    """Diverging ranked-correlation bar; optional muted reference bar on top."""
    names = list(labels)
    values = list(vals)
    colors = [NEG if v < 0 else POS for v in values]
    if ref_label is not None:
        names = names + [ref_label]
        values = values + [ref_val]
        colors = colors + [ACCENT]
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, height=0.6, zorder=3)
    ax.axvline(0, color=INK, lw=1)
    lim = max(0.6, max(abs(v) for v in values) + 0.12)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK, pad=8, loc="left")
    for yi, name, v in zip(y, names, values):
        ax.text(-0.02 * lim / 0.6 if v >= 0 else 0.02 * lim / 0.6, yi, name, va="center",
                ha="right" if v >= 0 else "left", fontsize=10, color=INK)
        ax.text(v + (0.02 if v >= 0 else -0.02) * lim / 0.6, yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5, fontweight="bold", color=INK_2)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def fig_diversity_vs_price(per):
    names = list(SIMPSON_BLOCKS)
    corr = {n: float(np.corrcoef(per[n], per["median_price"])[0, 1]) for n in names}
    order = sorted(names, key=lambda n: abs(corr[n]), reverse=True)

    fig = plt.figure(figsize=(14, 8.6))
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(
        2, 3, figure=fig, left=0.065, right=0.975, top=0.84, bottom=0.10, hspace=0.42, wspace=0.28
    )

    price = per["median_price"].to_numpy()
    for i, name in enumerate(order):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        _scatter(ax, per[name].to_numpy(), price, name)
        if i % 3 == 0:
            ax.set_ylabel("Median price", fontsize=10, color=INK_2)
        ax.set_xlabel("Simpson diversity  (0 = uniform, 1 = mixed)", fontsize=9.5, color=INK_2)

    # sixth cell: ranked |r| summary bar
    ax = fig.add_subplot(gs[1, 2])
    _corr_summary_bar(ax, order[::-1], [corr[n] for n in order[::-1]], title="Correlation with price")

    title_block(
        fig,
        "Neighbourhood Diversity vs House Price",
        "Simpson (Gini-Simpson) diversity of each social block against median price, one point per MSOA  ·  "
        "only commute-distance mix shows a real link",
    )
    footer(
        fig,
        "Simpson index per MSOA (43 areas)  ·  Pearson r vs median sale price  ·  "
        "Shannon entropy gives the same ranking",
    )
    out = IMAGE_DIR / "msoa_diversity_vs_price.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 3: diversity vs model fit (per-MSOA R2) -----------------------------
def fig_diversity_vs_r2(per, r2):
    p = per.copy()
    p["r2"] = p.index.map(r2)
    p = p[p["r2"].notna()]
    names = list(SIMPSON_BLOCKS)
    corr = {n: float(np.corrcoef(p[n], p["r2"])[0, 1]) for n in names}
    order = sorted(names, key=lambda n: abs(corr[n]), reverse=True)
    n_r = float(np.corrcoef(p["n_sales"], p["r2"])[0, 1])

    fig = plt.figure(figsize=(14, 8.6))
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(
        2, 3, figure=fig, left=0.065, right=0.975, top=0.84, bottom=0.10, hspace=0.42, wspace=0.28
    )

    r2v = p["r2"].to_numpy()
    yfmt = lambda v, _: f"{v:.2f}"  # noqa: E731
    for i, name in enumerate(order):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        _scatter(ax, p[name].to_numpy(), r2v, name, yfmt=yfmt)
        if i % 3 == 0:
            ax.set_ylabel("Test R²", fontsize=10, color=INK_2)
        ax.set_xlabel("Simpson diversity  (0 = uniform, 1 = mixed)", fontsize=9.5, color=INK_2)

    ax = fig.add_subplot(gs[1, 2])
    _corr_summary_bar(
        ax,
        order[::-1],
        [corr[n] for n in order[::-1]],
        ref_label="Sales per area (n)",
        ref_val=n_r,
        title="Correlation with model R²",
    )

    title_block(
        fig,
        "Does Diversity Explain Where the Model Struggles?",
        "Simpson diversity of each social block vs the model's test R² per MSOA  ·  "
        "no block explains the fit — sample size does",
    )
    footer(
        fig,
        "Test R² per MSOA (full features + Cambridge/London distance, 43 areas)  ·  Pearson r  ·  "
        f"orange = number of sales, the real driver of fit (r={n_r:+.2f})",
    )
    out = IMAGE_DIR / "msoa_diversity_vs_model_fit.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    df, gdf = load()
    r2 = parse_regional_r2()
    per = per_msoa_diversity(df)

    outputs = [
        fig_r2_map(gdf, r2),
        fig_diversity_vs_price(per),
        fig_diversity_vs_r2(per, r2),
    ]
    for out in outputs:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
