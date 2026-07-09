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
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, NullLocator

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
    TEAL_RAMP,
    STAR,
    load,
    title_block,
    footer,
    _bare,
    _project,
)

# best/worst highlight colours + panel surface
GOOD = "#0e6f4e"
BAD = "#d4380d"
PANEL = "#f2f1ec"

R2_RESULT = (
    PROJECT_ROOT / "outputs" / "msoa_price_map" / "result of RF for different region.txt"
)
MSOA_NAMES = PROJECT_ROOT / "data" / "spatial" / "msoa_names.csv"
PTYPE_DIVERSITY_CSV = PROJECT_ROOT / "data" / "spatial" / "msoa_property_type_diversity.csv"
IMPORTANCE_CSV = PROJECT_ROOT / "outputs" / "msoa_price_map" / "rf_cambridge_london_feature_importances.csv"
MODEL_NAME = "RF Full Features + Cambridge/London Distance"

# feature-type colours (Property/Location/Social match the correlation-rank chart;
# Time is added because the model also uses the sale date)
TYPE_COLOR = {"Property": "#2a78d6", "Location": "#1baf7a", "Time": "#4a3aa7", "Social": "#c9c7bf"}

# raw model feature (prefix stripped) -> (display name, feature type)
FEATURE_META = {
    "num_bed_": ("Bedrooms", "Property"),
    "num_bath": ("Bathrooms", "Property"),
    "num_reception": ("Receptions", "Property"),
    "distance_to_cambridge_km": ("Distance to Cambridge", "Location"),
    "distance_to_london_km": ("Distance to London", "Location"),
    "start_year": ("Sale year", "Time"),
    "start_month": ("Sale month", "Time"),
    "Cycle": ("Cycle to work", "Social"),
    "Driving": ("Drives to work", "Social"),
    "Foot": ("Walks to work", "Social"),
    "Rail": ("Rail to work", "Social"),
    "City Public": ("Public transport", "Social"),
    "Other Method": ("Other commute", "Social"),
    "Low": ("Low qualifications", "Social"),
    "Medium": ("Medium qualifications", "Social"),
    "High": ("Degree-level qualifications", "Social"),
    "Other Qualification": ("Other qualifications", "Social"),
    "Asian": ("Asian", "Social"),
    "Black": ("Black", "Social"),
    "White": ("White", "Social"),
    "Mixed": ("Mixed", "Social"),
    "Other": ("Other ethnicity", "Social"),
    "Minors 0-18": ("Under-18s", "Social"),
    "Adults 18-60": ("Adults 18-60", "Social"),
    "Elders >60": ("Over-60s", "Social"),
    "0-5km": ("Commute 0-5km", "Social"),
    "5-30km": ("Commute 5-30km", "Social"),
    ">30km": ("Commute >30km", "Social"),
    "Work from Home (0km)": ("Work from home", "Social"),
    "Living Offshore": ("Offshore workers", "Social"),
}

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


def shannon_index(block):
    """Shannon entropy, normalised to [0, 1] (H / ln K), on row-summed proportions."""
    p = block.div(block.sum(axis=1), axis=0).to_numpy(float)
    k = p.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return pd.Series(-terms.sum(axis=1) / np.log(k), index=block.index)


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

    # size the figure to the region's own aspect ratio so there is no dead space
    minx, miny, maxx, maxy = g.total_bounds
    data_ar = (maxy - miny) / (maxx - minx)
    fig_w, map_x0, map_wf = 12.5, 0.14, 0.85
    top_in, bot_in = 1.6, 0.55
    map_h_in = map_wf * fig_w * data_ar
    fig_h = map_h_in + top_in + bot_in
    map_y0, map_hf = bot_in / fig_h, map_h_in / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([map_x0, map_y0, map_wf, map_hf])
    norm = Normalize(g["r2"].min(), g["r2"].max())
    g.plot(ax=ax, column="r2", cmap=BLUE_RAMP, norm=norm, linewidth=0.6, edgecolor="white")
    ax.set_axis_off()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")

    best3.plot(ax=ax, facecolor="none", edgecolor=GOOD, linewidth=2.8, zorder=5)
    worst3.plot(ax=ax, facecolor="none", edgecolor=BAD, linewidth=2.8, zorder=5)

    # label the 6 highlighted areas, pushed radially outward to avoid the centre;
    # Eddington is nudged to the top-left so it clears the Isaac Newton star
    halo = [path_effects.withStroke(linewidth=3, foreground=SURFACE)]
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2
    for rank_set, color in ((best3, GOOD), (worst3, BAD)):
        for rank, (_, row) in enumerate(rank_set.iterrows(), start=1):
            if "Eddington" in row["name"]:
                off_pt, ha, va = (-118, 82), "right", "center"
            else:
                vx, vy = row["cx"] - mx, row["cy"] - my
                dist = np.hypot(vx, vy) or 1.0
                ux, uy = vx / dist, vy / dist
                off_pt = (ux * 82, uy * 82)
                ha, va = ("left" if ux >= 0 else "right"), ("bottom" if uy >= 0 else "top")
            ax.plot(row["cx"], row["cy"], "o", ms=6, mfc=color, mec="white", mew=1.1, zorder=9)
            ax.annotate(
                f"{rank}. {row['name']}", (row["cx"], row["cy"]),
                xytext=off_pt, textcoords="offset points", ha=ha, va=va,
                fontsize=11, fontweight="bold", color=color, zorder=10,
                path_effects=halo,
                arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=0, shrinkB=6),
            )

    # Isaac Newton Institute — the headline landmark, "where we are"; points down-right
    newton = _project([("Isaac Newton Institute", 52.2109, 0.0985)]).geometry.iloc[0]
    ax.plot(newton.x, newton.y, marker="*", ms=28, mfc=STAR, mec="white", mew=1.7, zorder=11)
    ax.annotate(
        "Isaac Newton Institute\n(where we are)", (newton.x, newton.y),
        xytext=(74, -66), textcoords="offset points", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color=STAR, zorder=12, path_effects=halo,
        arrowprops=dict(arrowstyle="-", color=STAR, lw=1.3, shrinkA=0, shrinkB=9),
    )

    # vertical colorbar on the left, centred against the map
    cax_hf = 0.46 * map_hf
    cax = fig.add_axes([0.055, map_y0 + (map_hf - cax_hf) / 2, 0.026, cax_hf])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=BLUE_RAMP, norm=norm), cax=cax, orientation="vertical")
    cb.set_label("Test R²  (price scale)", color=INK, fontsize=12.5, fontweight="bold")
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    cb.ax.tick_params(colors=MUTED, labelsize=10, length=0)
    cax.text(0.5, 1.04, "better fit", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=9.5, color=GOOD, fontweight="bold")
    cax.text(0.5, -0.04, "weaker fit", transform=cax.transAxes, ha="center", va="top",
             fontsize=9.5, color=BAD, fontweight="bold")

    title_block(
        fig,
        "How Well the Model Predicts, by Area",
        "Random Forest test R² per MSOA — full features + distance to Cambridge & London  ·  darker = better fit",
    )
    footer(
        fig,
        "R² on a 20% hold-out set, per MSOA  ·  ★ Isaac Newton Institute marks where we are  ·  "
        "Boundaries: ONS MSOA 2021  ·  EPSG:27700",
    )
    out = IMAGE_DIR / "rf_full_features_msoa_r2_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_r2_comparison(table, names):
    t = table.copy()
    t["name"] = names.reindex(t.index).fillna(pd.Series(t.index, index=t.index))
    order = t.sort_values("r2_price", ascending=False)
    best3, worst3 = order.head(3), order.tail(3).iloc[::-1]

    def rows_of(frame):
        return [(r["name"], r["r2_price"], int(r["n_test"]), float(r["MAPE"]))
                for _, r in frame.iterrows()]

    def avg_of(frame):
        return (frame["r2_price"].mean(), frame["n_test"].mean(), frame["MAPE"].mean())

    fig = plt.figure(figsize=(13, 5.0))
    fig.patch.set_facecolor(SURFACE)
    pax = fig.add_axes([0.04, 0.05, 0.92, 0.72])
    pax.set_xlim(0, 1)
    pax.set_ylim(0, 1)
    pax.axis("off")
    _group_panel(pax, 0.0, 0.47, "Best-fit areas", GOOD,
                 "the model nails these", rows_of(best3), avg_of(best3))
    _group_panel(pax, 0.53, 0.47, "Worst-fit areas", BAD,
                 "the model struggles here", rows_of(worst3), avg_of(worst3))

    title_block(
        fig,
        "Best- and Worst-fit Areas",
        "per-MSOA test R² — full features + Cambridge/London distance  ·  worst-fit areas have far fewer "
        "sales and larger % errors",
    )
    footer(fig, "R² on a 20% hold-out set, per MSOA  ·  sales = hold-out sales scored per area")
    out = IMAGE_DIR / "rf_full_features_msoa_r2_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 2: diversity vs price ----------------------------------------------
def _fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _scatter(ax, x, y, name, yfmt=lambda v, _: f"£{v/1000:.0f}k", show_stats=True):
    r, p = pearsonr(x, y)
    ax.scatter(x, y, s=34, color=POS, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color=INK, lw=1.4, ls="--", zorder=4)
    ax.set_title(name, fontsize=12.5, fontweight="bold", color=INK, pad=8, loc="left")
    if show_stats:
        ax.text(
            0.04, 0.14, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=NEG if r < 0 else POS, ha="left", va="bottom",
        )
        ax.text(
            0.04, 0.05, _fmt_p(p) + ("" if p < 0.05 else "  (n.s.)"), transform=ax.transAxes,
            fontsize=9.5, color=INK_2 if p < 0.05 else MUTED, ha="left", va="bottom",
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


def _r_p_table(ax, rows, title):
    """Small, uncoloured r / p comparison table. rows = [(label, r, p), ...]."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK, pad=8, loc="left")
    xr, xp = 0.72, 1.0
    ax.text(0.0, 0.9, "Feature", fontsize=9.5, color=MUTED, ha="left", va="center")
    ax.text(xr, 0.9, "r", fontsize=9.5, color=MUTED, ha="right", va="center")
    ax.text(xp, 0.9, "p", fontsize=9.5, color=MUTED, ha="right", va="center")
    ax.plot([0, 1], [0.83, 0.83], color=GRID, lw=1)
    y = 0.68
    for label, r, pval in rows:
        ax.text(0.0, y, label, fontsize=10.5, color=INK, ha="left", va="center")
        ax.text(xr, y, f"{r:+.2f}", fontsize=10.5, color=INK, ha="right", va="center")
        ptxt = "<0.001" if pval < 0.001 else f"{pval:.3f}"
        ax.text(xp, y, ptxt, fontsize=10.5, color=INK_2, ha="right", va="center")
        y -= 0.15


# --- figure 3: diversity vs model fit (per-MSOA R2) -----------------------------
def fig_diversity_vs_r2(per, r2):
    p = per.copy()
    p["r2"] = p.index.map(r2)
    p = p[p["r2"].notna()]
    r2v = p["r2"].to_numpy(float)
    # Shannon diversity per block (normalised 0-1), for consistency with the deck
    shan = {n: shannon_index(p[cols]).to_numpy(float) for n, cols in SIMPSON_BLOCKS.items()}
    order = ["Ethnicity", "Age", "Education", "Commute distance", "Commute method"]
    stats = {n: pearsonr(shan[n], r2v) for n in order}  # (r, p)
    n_stat = pearsonr(p["n_sales"].to_numpy(float), r2v)

    fig = plt.figure(figsize=(14, 8.6))
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(
        2, 3, figure=fig, left=0.065, right=0.975, top=0.84, bottom=0.10, hspace=0.42, wspace=0.28
    )

    yfmt = lambda v, _: f"{v:.2f}"  # noqa: E731
    for i, name in enumerate(order):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        _scatter(ax, shan[name], r2v, name, yfmt=yfmt, show_stats=False)
        if i % 3 == 0:
            ax.set_ylabel("Test R²", fontsize=10, color=INK_2)
        ax.set_xlabel("Shannon diversity", fontsize=9.5, color=INK_2)

    ax = fig.add_subplot(gs[1, 2])
    table_rows = [(n, stats[n][0], stats[n][1]) for n in order]
    _r_p_table(ax, table_rows, "Correlation with model R²")

    title_block(
        fig,
        "Does Diversity Explain Where the Model Struggles?",
        "Shannon diversity of each social block vs the model's test R² per MSOA  ·  "
        "no block explains the fit — sample size does",
    )
    footer(
        fig,
        "Test R² per MSOA (full features + Cambridge/London distance, 43 areas)  ·  Pearson r & p  ·  "
        f"no block is significant — the real driver of fit is sample size (r={n_stat[0]:+.2f})",
    )
    out = IMAGE_DIR / "msoa_diversity_vs_model_fit.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 4: data volume vs model fit -----------------------------------------
def fig_sales_vs_r2(table, names):
    d = table.copy()
    d["r2"] = d["r2_price"]
    d["name"] = names.reindex(d.index).fillna(pd.Series(d.index, index=d.index))
    x = d["n_test"].to_numpy(float)
    y = d["r2"].to_numpy(float)
    r, p = pearsonr(x, y)

    order = d.sort_values("r2")
    worst3 = set(order.head(3).index)
    best3 = set(order.tail(3).index)

    fig = plt.figure(figsize=(14, 6.6))
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1],
                  left=0.07, right=0.975, top=0.80, bottom=0.13, wspace=0.22)

    # scatter: sales vs R2
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    colors = [BAD if i in worst3 else GOOD if i in best3 else POS for i in d.index]
    ax.scatter(x, y, s=52, c=colors, alpha=0.82, edgecolor="white", linewidth=0.6, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color=INK, lw=1.6, ls="--", zorder=4)
    ax.text(0.035, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=POS, va="top")
    ax.text(0.035, 0.885, _fmt_p(p), transform=ax.transAxes, fontsize=10.5, color=INK_2, va="top")
    halo = [path_effects.withStroke(linewidth=2.6, foreground=SURFACE)]
    for i in worst3:
        row = d.loc[i]
        ax.annotate(row["name"], (row["n_test"], row["r2"]), xytext=(8, 0),
                    textcoords="offset points", fontsize=8.6, color=BAD, va="center", ha="left",
                    fontweight="bold", path_effects=halo)
    ax.set_xlabel("Sales scored per area (hold-out set)", fontsize=10.5, color=INK_2)
    ax.set_ylabel("Test R²", fontsize=10.5, color=INK_2)
    _bare(ax)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)
    ax.set_title("More sales → better, steadier fit", fontsize=13, fontweight="bold",
                 color=INK, loc="left", pad=8)
    handles = [
        Patch(color=GOOD, label="best-fit 3"),
        Patch(color=BAD, label="worst-fit 3"),
        Patch(color=POS, label="other areas"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_2)

    # tier bars: average R2 by data volume
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(SURFACE)
    edges = [0, 150, 250, 400, np.inf]
    labs = ["<150", "150–250", "250–400", "400+"]
    d["tier"] = pd.cut(d["n_test"], bins=edges, labels=labs, right=False)
    grp = d.groupby("tier", observed=False)["r2"].agg(["mean", "size"]).reindex(labs)
    yb = np.arange(len(labs))
    ramp = [TEAL_RAMP(0.30 + 0.6 * i / (len(labs) - 1)) for i in range(len(labs))]
    ax2.bar(yb, grp["mean"], color=ramp, width=0.68, zorder=3)
    for i, (m, nn) in enumerate(zip(grp["mean"], grp["size"])):
        if np.isfinite(m):
            ax2.text(i, m + 0.008, f"{m:.2f}", ha="center", va="bottom", fontsize=11.5,
                     fontweight="bold", color=INK)
            ax2.text(i, 0.03, f"{int(nn)} areas", ha="center", va="bottom", fontsize=9,
                     color="white", fontweight="bold", zorder=4)
    ax2.set_xticks(yb)
    ax2.set_xticklabels(labs, fontsize=10, color=INK)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Sales scored per area", fontsize=10.5, color=INK_2)
    ax2.set_ylabel("Average test R²", fontsize=10.5, color=INK_2)
    _bare(ax2)
    ax2.set_title("Average fit by data volume", fontsize=13, fontweight="bold",
                  color=INK, loc="left", pad=8)

    title_block(
        fig,
        "Why Some Areas Fit Better: Data Volume",
        "each area's hold-out R² against how many sales it has  ·  thin-data areas fit worst and noisiest",
    )
    footer(
        fig,
        "43 MSOAs  ·  Pearson r  ·  worst-fit areas (red) sit at low sales counts; more data lifts and steadies R²",
    )
    out = IMAGE_DIR / "msoa_sales_vs_r2.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 5: Gini feature-importance ranking ----------------------------------
def load_importances(model=MODEL_NAME):
    d = pd.read_csv(IMPORTANCE_CSV)
    d = d[d["model"] == model]
    agg = {}
    for feat, imp in zip(d["feature"], d["importance"]):
        name = feat.split("__", 1)[1] if "__" in feat else feat
        if name.startswith("property_type_model_"):
            disp, typ = "Property type", "Property"
        else:
            disp, typ = FEATURE_META.get(name, (name, "Social"))
        val, _ = agg.get(disp, (0.0, typ))
        agg[disp] = (val + float(imp), typ)
    rows = [(disp, typ, val) for disp, (val, typ) in agg.items()]
    rows.sort(key=lambda t: t[2])  # ascending -> largest bar on top
    return rows


def fig_gini_importance_rank():
    rows = load_importances()
    labels = [r[0] for r in rows]
    types = [r[1] for r in rows]
    vals = [r[2] for r in rows]
    colors = [TYPE_COLOR[t] for t in types]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(12, 11))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.33, right=0.93, top=0.855, bottom=0.07)
    ax.set_facecolor(SURFACE)
    ax.barh(y, vals, color=colors, height=0.76, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.008, i, f"{v*100:.1f}%", va="center", ha="left",
                fontsize=8.8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.4, color=INK)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    _bare(ax)
    ax.set_xlabel("Random Forest importance  (share of split reduction, Gini)", fontsize=10, color=INK_2)

    present = [t for t in ("Property", "Location", "Time", "Social") if t in types]
    leg = ax.legend(
        handles=[Patch(color=TYPE_COLOR[c], label=c) for c in present],
        loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK_2,
        title="Feature type", title_fontsize=10,
    )
    leg.get_title().set_color(INK)
    title_block(
        fig,
        "What the Model Leans On Most",
        "every feature ranked by Random Forest importance (Gini)  ·  property-type categories combined",
    )
    footer(
        fig,
        "Gini importance sums to 100% across features  ·  full-features model  ·  "
        "bedrooms dominate; the two distances rank in the top group",
    )
    out = IMAGE_DIR / "rf_full_features_gini_importance_rank.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 6: property-type diversity vs model fit -----------------------------
def _partial_r(a, b, c):
    """Partial correlation of a and b, controlling for c."""
    rac = pearsonr(a, c)[0]
    rbc = pearsonr(b, c)[0]
    rab = pearsonr(a, b)[0]
    return (rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2))


def fig_ptype_diversity_vs_r2(table, names):
    div = pd.read_csv(PTYPE_DIVERSITY_CSV).set_index("msoa21")
    d = div.join(table[["r2_price", "n_test"]]).dropna()
    d["name"] = names.reindex(d.index).fillna(pd.Series(d.index, index=d.index))
    x = d["shannon_entropy"].to_numpy(float)
    y = d["r2_price"].to_numpy(float)
    r, p = pearsonr(x, y)
    pr = _partial_r(x, y, d["n_test"].to_numpy(float))

    order = d.sort_values("r2_price")
    worst3 = set(order.head(3).index)
    best3 = set(order.tail(3).index)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.09, right=0.965, top=0.82, bottom=0.11)

    colors = [BAD if i in worst3 else GOOD if i in best3 else POS for i in d.index]
    ax.scatter(x, y, s=62, c=colors, alpha=0.82, edgecolor="white", linewidth=0.7, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color=INK, lw=1.6, ls="--", zorder=4)

    ax.text(0.03, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=POS if r >= 0 else NEG, va="top")
    ax.text(0.03, 0.885, _fmt_p(p) + ("" if p < 0.05 else "  (n.s.)"),
            transform=ax.transAxes, fontsize=11, color=INK_2, va="top")
    ax.text(0.03, 0.825, f"controlling for sales count:  r = {pr:+.2f}",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="top")

    ax.set_xlabel("Property-type diversity  (Shannon entropy, nats)", fontsize=11, color=INK_2)
    ax.set_ylabel("Test R²", fontsize=11, color=INK_2)
    _bare(ax)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)
    ax.legend(
        handles=[Patch(color=GOOD, label="best-fit 3"), Patch(color=BAD, label="worst-fit 3"),
                 Patch(color=POS, label="other areas")],
        loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK_2,
    )
    title_block(
        fig,
        "Property-type Diversity vs Model Fit",
        "each area's property-type Shannon entropy against the model's test R²  ·  no real link",
    )
    footer(
        fig,
        "43 MSOAs  ·  Pearson r vs R²  ·  the weak raw link disappears once sales count is held fixed (partial r ≈ 0)",
    )
    out = IMAGE_DIR / "msoa_ptype_diversity_vs_r2.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure: repeat price & room-type sales vs model fit ------------------------
SHARED_KEYS = ["price_sold", "property_type", "num_bed_", "num_bath", "num_reception"]


def fig_duplication_vs_r2(df, table, names):
    # separate sales that share an identical price + room-type profile with another
    shared = df.groupby(SHARED_KEYS)["price_sold"].transform("size") >= 2
    cnt = df.assign(_s=shared.values).groupby("msoa21")["_s"].sum()
    d = pd.DataFrame({"shared": cnt}).join(table[["r2_price", "n_test"]]).dropna()
    d["name"] = names.reindex(d.index).fillna(pd.Series(d.index, index=d.index))
    x = d["shared"].to_numpy(float)
    y = d["r2_price"].to_numpy(float)
    lx = np.log(x)
    r, p = pearsonr(lx, y)
    pr = _partial_r(lx, y, np.log(d["n_test"].to_numpy(float)))

    order = d.sort_values("r2_price")
    worst3 = set(order.head(3).index)
    best3 = set(order.tail(3).index)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.10, right=0.965, top=0.82, bottom=0.11)

    colors = [BAD if i in worst3 else GOOD if i in best3 else POS for i in d.index]
    ax.scatter(x, y, s=62, c=colors, alpha=0.82, edgecolor="white", linewidth=0.7, zorder=3)
    slope, intercept = np.polyfit(lx, y, 1)
    xs = np.geomspace(x.min(), x.max(), 60)
    ax.plot(xs, slope * np.log(xs) + intercept, color=INK, lw=1.6, ls="--", zorder=4)

    ax.text(0.03, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=POS if r >= 0 else NEG, va="top")
    ax.text(0.03, 0.885, _fmt_p(p) + ("" if p < 0.05 else "  (n.s.)"),
            transform=ax.transAxes, fontsize=11, color=INK_2, va="top")

    ax.set_xlabel("Sales sharing an identical price & room type  (count, log scale)",
                  fontsize=11.5, color=INK_2)
    ax.set_ylabel("Test R²", fontsize=11.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([200, 300, 500, 1000, 2000, 3000])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _bare(ax)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)
    ax.legend(
        handles=[Patch(color=GOOD, label="best-fit 3"), Patch(color=BAD, label="worst-fit 3"),
                 Patch(color=POS, label="other areas")],
        loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_2,
    )

    title_block(
        fig,
        "Repeat Price & Room-type Sales vs Model Fit",
        "count of separate sales that share an identical price and room type vs test R²  ·  "
        "different buyers, not data errors",
    )
    footer(
        fig,
        "43 MSOAs  ·  Pearson r vs R²  ·  these areas are mostly just the larger, more liquid ones "
        f"(r={pr:+.2f} once sales count is held fixed)  ·  identical price+room rows split across train & "
        "test still let the model memorise",
    )
    out = IMAGE_DIR / "msoa_duplication_vs_r2.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --- figure 7: each area as a point in the sales x diversity plane, coloured by R2
# red-grey-blue ramp: keeps low-R2 points visible (they read warm, not washed out)
FIT_RAMP = LinearSegmentedColormap.from_list(
    "fit", ["#c0392b", "#e08a5a", "#b8b5ad", "#5b9bd5", "#123f7a"]
)


def fig_fit_by_sales_diversity(table, names):
    div = pd.read_csv(PTYPE_DIVERSITY_CSV).set_index("msoa21")
    d = div.join(table[["r2_price", "n_test"]]).dropna()
    d["name"] = names.reindex(d.index).fillna(pd.Series(d.index, index=d.index))
    x = d["n_test"].to_numpy(float)
    y = d["shannon_entropy"].to_numpy(float)
    r2 = d["r2_price"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(11.5, 8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.11)

    norm = Normalize(r2.min(), r2.max())
    sc = ax.scatter(x, y, c=r2, cmap=FIT_RAMP, norm=norm, s=190,
                    edgecolor="white", linewidth=1.1, zorder=3)

    # label the three lowest-R2 areas (the reddest), offset to avoid overlap
    worst = d.sort_values("r2_price").head(3)
    offs = [(14, 12), (14, -16), (16, 30)]  # Eddington, Milton, Barrington
    halo = [path_effects.withStroke(linewidth=2.6, foreground=SURFACE)]
    for k, (_, row) in enumerate(worst.iterrows()):
        ax.annotate(f"{row['name']}  (R² {row['r2_price']:.2f})",
                    (row["n_test"], row["shannon_entropy"]), xytext=offs[k],
                    textcoords="offset points", fontsize=9, color="#8f1d13", fontweight="bold",
                    ha="left", va="center", path_effects=halo,
                    arrowprops=dict(arrowstyle="-", color="#8f1d13", lw=1))

    ax.set_xlabel("Sales per area  —  data volume  →", fontsize=11.5, color=INK_2)
    ax.set_ylabel("Property-type diversity  (Shannon H)  →", fontsize=11.5, color=INK_2)
    _bare(ax)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)

    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Test R²", color=INK_2, fontsize=10.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0, labelsize=9)

    fig.text(0.055, 0.945, "Model Fit by Data Volume and Diversity", fontsize=19,
             fontweight="bold", color=INK)
    fig.text(0.055, 0.905,
             "each area placed by its sales count and property-type diversity, coloured by test R²  ·  "
             "red (weak fit) lines up on the left", fontsize=11.5, color=INK_2)
    fig.text(0.055, 0.03,
             "43 MSOAs  ·  R² vs log(sales) r=+0.65; diversity adds ~nothing once sales is known "
             "(partial r=+0.03) — red points span every diversity level, but only low sales", fontsize=9, color=MUTED)

    out = IMAGE_DIR / "msoa_fit_by_sales_diversity.png"
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
        fig_r2_comparison(table, names),
        fig_diversity_vs_price(per),
        fig_diversity_vs_r2(per, table["r2_price"]),
        fig_sales_vs_r2(table, names),
        fig_gini_importance_rank(),
        fig_ptype_diversity_vs_r2(table, names),
        fig_duplication_vs_r2(df, table, names),
        fig_fit_by_sales_diversity(table, names),
    ]
    for out in outputs:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
