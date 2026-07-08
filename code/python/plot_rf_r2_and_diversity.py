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
from scipy.stats import pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

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
    title_block,
    footer,
)

# best/worst highlight colours + panel surface
GOOD = "#0e6f4e"
BAD = "#d4380d"
PANEL = "#f2f1ec"

R2_RESULT = (
    PROJECT_ROOT / "outputs" / "msoa_price_map" / "result of RF for different region.txt"
)
MSOA_NAMES = PROJECT_ROOT / "data" / "spatial" / "msoa_names.csv"
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


def parse_regional_table(path=R2_RESULT, model_name=MODEL_NAME):
    """Per-MSOA regional metrics for one model, as a DataFrame indexed by msoa21."""
    rows = []
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
            f = stripped.split()
            rows.append(
                {
                    "msoa21": f[0],
                    "n_test": int(f[1]),
                    "actual_mean": int(f[2].replace(",", "")),
                    "MAE": int(f[4].replace(",", "")),
                    "RMSE": int(f[5].replace(",", "")),
                    "MAPE": float(f[6]),
                    "r2_price": float(f[-1]),
                }
            )
    if not rows:
        raise ValueError(f"No rows parsed for model '{model_name}' in {path}")
    return pd.DataFrame(rows).set_index("msoa21")


def parse_regional_r2(path=R2_RESULT, model_name=MODEL_NAME):
    """Per-MSOA R2 (price scale) for one model, as a Series."""
    return parse_regional_table(path, model_name)["r2_price"]


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
def _group_panel(pax, x0, w, header, color, sub, rows, avg):
    """One comparison column (best or worst) inside the bottom panel axes."""
    pax.add_patch(
        FancyBboxPatch(
            (x0, 0.0), w, 1.0, boxstyle="round,pad=0.006,rounding_size=0.03",
            fc=PANEL, ec="none", transform=pax.transAxes, zorder=1,
        )
    )
    pad = 0.03
    pax.text(x0 + pad, 0.88, header, color=color, fontsize=15, fontweight="bold",
             transform=pax.transAxes, va="center")
    pax.text(x0 + pad, 0.79, sub, color=MUTED, fontsize=10.5, transform=pax.transAxes, va="center")
    y = 0.63
    for i, (name, r2v, n, mape) in enumerate(rows, start=1):
        pax.text(x0 + pad, y, f"{i}", color=color, fontsize=13, fontweight="bold",
                 transform=pax.transAxes, ha="left", va="center")
        pax.text(x0 + pad + 0.032, y, name, color=INK, fontsize=12.5, transform=pax.transAxes, va="center")
        pax.text(x0 + pad + 0.032, y - 0.08,
                 f"R² {r2v:.2f}    ·    {n:,} sales    ·    {mape:.1f}% error",
                 color=INK_2, fontsize=10.5, transform=pax.transAxes, va="center")
        y -= 0.18
    pax.plot([x0 + pad, x0 + w - pad], [y + 0.03, y + 0.03], color=GRID, lw=1, transform=pax.transAxes)
    pax.text(x0 + pad, y - 0.03, "Group average", color=MUTED, fontsize=10.5,
             transform=pax.transAxes, va="center")
    pax.text(x0 + w - pad, y - 0.03,
             f"R² {avg[0]:.2f}   ·   {avg[1]:,.0f} sales   ·   {avg[2]:.1f}% error",
             color=color, fontsize=11.5, fontweight="bold", transform=pax.transAxes, ha="right", va="center")


def fig_r2_map(gdf, table, names):
    g = gdf.copy()
    g["r2"] = g["MSOA21CD"].map(table["r2_price"])
    if g["r2"].isna().any():
        raise ValueError("some MSOAs on the map have no R2 value")
    g["name"] = g["MSOA21CD"].map(names).fillna(g["MSOA21CD"])
    g["cx"] = g.geometry.centroid.x
    g["cy"] = g.geometry.centroid.y

    order = g.sort_values("r2", ascending=False)
    best3 = order.head(3)
    worst3 = order.tail(3).iloc[::-1]  # worst first

    fig = plt.figure(figsize=(12.5, 15.4))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.13, 0.325, 0.85, 0.60])
    norm = Normalize(g["r2"].min(), g["r2"].max())
    g.plot(ax=ax, column="r2", cmap=BLUE_RAMP, norm=norm, linewidth=0.6, edgecolor="white")
    ax.set_axis_off()
    ax.set_aspect("equal")

    best3.plot(ax=ax, facecolor="none", edgecolor=GOOD, linewidth=2.8, zorder=5)
    worst3.plot(ax=ax, facecolor="none", edgecolor=BAD, linewidth=2.8, zorder=5)

    # label the 6 highlighted areas, pushed radially outward to avoid the centre
    halo = [path_effects.withStroke(linewidth=3, foreground=SURFACE)]
    minx, miny, maxx, maxy = g.total_bounds
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2
    for rank_set, color in ((best3, GOOD), (worst3, BAD)):
        for rank, (_, row) in enumerate(rank_set.iterrows(), start=1):
            vx, vy = row["cx"] - mx, row["cy"] - my
            dist = np.hypot(vx, vy) or 1.0
            ux, uy = vx / dist, vy / dist
            off = 82
            ax.plot(row["cx"], row["cy"], "o", ms=6, mfc=color, mec="white", mew=1.1, zorder=9)
            ax.annotate(
                f"{rank}. {row['name']}",
                (row["cx"], row["cy"]),
                xytext=(ux * off, uy * off),
                textcoords="offset points",
                ha="left" if ux >= 0 else "right",
                va="bottom" if uy >= 0 else "top",
                fontsize=11,
                fontweight="bold",
                color=color,
                zorder=10,
                path_effects=halo,
                arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=0, shrinkB=6),
            )

    # vertical colorbar on the left
    cax = fig.add_axes([0.055, 0.47, 0.026, 0.34])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=BLUE_RAMP, norm=norm), cax=cax, orientation="vertical")
    cb.set_label("Test R²  (price scale)", color=INK, fontsize=12.5, fontweight="bold")
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    cb.ax.tick_params(colors=MUTED, labelsize=10, length=0)
    cax.text(0.5, 1.04, "better fit", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=9.5, color=GOOD, fontweight="bold")
    cax.text(0.5, -0.04, "weaker fit", transform=cax.transAxes, ha="center", va="top",
             fontsize=9.5, color=BAD, fontweight="bold")

    # comparison panel below the map
    def rows_of(frame):
        return [
            (r["name"], r["r2"], int(table.loc[r["MSOA21CD"], "n_test"]),
             float(table.loc[r["MSOA21CD"], "MAPE"]))
            for _, r in frame.iterrows()
        ]

    def avg_of(frame):
        idx = frame["MSOA21CD"]
        return (frame["r2"].mean(), table.loc[idx, "n_test"].mean(), table.loc[idx, "MAPE"].mean())

    pax = fig.add_axes([0.04, 0.035, 0.92, 0.235])
    pax.set_xlim(0, 1)
    pax.set_ylim(0, 1)
    pax.axis("off")
    _group_panel(pax, 0.0, 0.47, "Best-fit areas", GOOD,
                 "the model nails these", rows_of(best3), avg_of(best3))
    _group_panel(pax, 0.53, 0.47, "Worst-fit areas", BAD,
                 "the model struggles here", rows_of(worst3), avg_of(worst3))

    title_block(
        fig,
        "How Well the Model Predicts, by Area",
        "Random Forest test R² per MSOA — full features + distance to Cambridge & London  ·  darker = better fit",
    )
    footer(
        fig,
        "R² on a 20% hold-out set, per MSOA  ·  worst-fit areas have far fewer sales and much larger % errors  ·  "
        "Boundaries: ONS MSOA 2021  ·  EPSG:27700",
    )
    out = IMAGE_DIR / "rf_full_features_msoa_r2_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 2: diversity vs price ----------------------------------------------
def _fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _scatter(ax, x, y, name, yfmt=lambda v, _: f"£{v/1000:.0f}k"):
    r, p = pearsonr(x, y)
    ax.scatter(x, y, s=34, color=POS, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color=INK, lw=1.4, ls="--", zorder=4)
    ax.set_title(name, fontsize=12.5, fontweight="bold", color=INK, pad=8, loc="left")
    ax.text(
        0.04,
        0.14,
        f"r = {r:+.2f}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=NEG if r < 0 else POS,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.04,
        0.05,
        _fmt_p(p) + ("" if p < 0.05 else "  (n.s.)"),
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK_2 if p < 0.05 else MUTED,
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
    table = parse_regional_table()
    names = pd.read_csv(MSOA_NAMES).set_index("msoa21")["name"]
    per = per_msoa_diversity(df)

    outputs = [
        fig_r2_map(gdf, table, names),
        fig_diversity_vs_price(per),
        fig_diversity_vs_r2(per, table["r2_price"]),
    ]
    for out in outputs:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
