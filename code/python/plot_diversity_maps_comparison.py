"""Diversity by dimension: Cambridgeshire vs Birmingham.

For each social dimension (ethnicity, age, qualifications, commute distance,
commute method) this makes a side-by-side normalised-Shannon diversity map of
the project's 43 Cambridgeshire MSOAs next to Birmingham's 132, on one shared
scale, plus a summary table of mean / min / max / SD per city.

Diversity values come from data/spatial/diversity_by_dimension_cam_bham.csv,
computed from Census 2021 tables (TS021, TS007A, TS067, TS058, TS061) via Nomis.
"""
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cambridgeshire_maps import (  # noqa: E402
    PROJECT_ROOT, IMAGE_DIR, BOUNDARIES,
    SURFACE, INK, INK_2, MUTED, GRID, VIOLET_RAMP,
)

DIV_CSV = PROJECT_ROOT / "data" / "spatial" / "diversity_by_dimension_cam_bham.csv"
PANEL = "#f2f1ec"

# key -> (display name, subtitle description, output filename stem, census table)
DIMS = {
    "ethnicity": ("Ethnic", "five ethnic groups", "ethnicity", "TS021"),
    "age": ("Age", "18 five-year age bands", "age", "TS007A"),
    "education": ("Qualification", "highest qualification, 7 levels", "education", "TS067"),
    "distance": ("Commute-distance", "distance travelled to work, 10 bands", "commute_distance", "TS058"),
    "method": ("Commute-method", "method of travel to work, 11 modes", "commute_method", "TS061"),
}


def load():
    div = pd.read_csv(DIV_CSV).set_index("msoa21")
    gdf = gpd.read_file(BOUNDARIES, columns=["MSOA21CD", "MSOA21NM", "geometry"])
    gdf = gdf[gdf["MSOA21CD"].isin(div.index)].to_crs(27700)
    gdf = gdf.merge(div, left_on="MSOA21CD", right_index=True)
    cam = gdf[gdf["city"] == "Cambridgeshire"]
    bham = gdf[gdf["city"] == "Birmingham"]
    return div, cam, bham


def make_map(cam, bham, key):
    name, desc, stem, table = DIMS[key]
    vmin = min(cam[key].min(), bham[key].min())
    vmax = max(cam[key].max(), bham[key].max())
    norm = Normalize(vmin, vmax)

    fig = plt.figure(figsize=(14, 8.6))
    fig.patch.set_facecolor(SURFACE)
    ax_c = fig.add_axes([0.02, 0.16, 0.46, 0.70])
    ax_b = fig.add_axes([0.50, 0.16, 0.46, 0.70])
    for ax, region, city in ((ax_c, cam, "Cambridgeshire"), (ax_b, bham, "Birmingham")):
        ax.set_facecolor(SURFACE)
        region.plot(ax=ax, column=key, cmap=VIOLET_RAMP, norm=norm, linewidth=0.4, edgecolor="white")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(city, fontsize=17, fontweight="bold", color=INK, pad=6)
        ax.text(0.5, -0.02, f"{len(region)} MSOAs   ·   avg {region[key].mean():.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=11.5, color=INK_2)

    cax = fig.add_axes([0.30, 0.085, 0.40, 0.02])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=VIOLET_RAMP, norm=norm), cax=cax, orientation="horizontal")
    cb.set_label(f"{name} diversity  —  Shannon index (0 = uniform, 1 = perfectly mixed)",
                 color=INK_2, fontsize=10.5)
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)

    fig.text(0.02, 0.955, f"{name} Diversity: Cambridgeshire vs Birmingham", fontsize=21,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.915,
             f"normalised Shannon diversity of {desc} per MSOA, one shared scale  ·  darker = more mixed",
             fontsize=12.5, color=INK_2)
    fig.text(0.02, 0.03,
             f"Source: ONS Census 2021, table {table}, via Nomis  ·  Boundaries: ONS MSOA 2021  ·  EPSG:27700",
             fontsize=9, color=MUTED)

    out = IMAGE_DIR / f"{stem}_diversity_cambridge_vs_birmingham.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_table(div):
    order = list(DIMS)
    g = div.groupby("city")[order].agg(["mean", "min", "max", "std"])

    fig = plt.figure(figsize=(13, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.74])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # column layout: label + Cambridgeshire(4) + Birmingham(4) + gap
    xc = [0.20, 0.30, 0.40, 0.50]          # Cambridgeshire mean/min/max/SD
    xb = [0.62, 0.72, 0.82, 0.92]          # Birmingham
    stat_names = ["mean", "min", "max", "SD"]
    stat_col = {"mean": "mean", "min": "min", "max": "max", "SD": "std"}

    ax.add_patch(FancyBboxPatch((0.155, 0.02), 0.415, 0.86, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc=PANEL, ec="none", transform=ax.transAxes, zorder=0))
    ax.add_patch(FancyBboxPatch((0.575, 0.02), 0.415, 0.86, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="#efeaf7", ec="none", transform=ax.transAxes, zorder=0))
    ax.text(0.36, 0.95, "Cambridgeshire", ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(0.78, 0.95, "Birmingham", ha="center", fontsize=13, fontweight="bold", color="#5a2d91")
    for x, s in zip(xc + xb, stat_names * 2):
        ax.text(x, 0.86, s, ha="center", fontsize=9.5, color=MUTED)
    ax.plot([0.0, 1.0], [0.81, 0.81], color=GRID, lw=1)

    y = 0.70
    for key in order:
        ax.text(0.0, y, DIMS[key][0].replace("-", " "), fontsize=12, color=INK, va="center")
        for x, s in zip(xc, stat_names):
            v = g.loc["Cambridgeshire", (key, stat_col[s])]
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=11, color=INK_2)
        for x, s in zip(xb, stat_names):
            v = g.loc["Birmingham", (key, stat_col[s])]
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=11, color=INK_2)
        y -= 0.135

    fig.text(0.02, 0.955, "Diversity by Dimension — Summary Statistics", fontsize=20,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.905,
             "normalised Shannon diversity per MSOA (0–1)  ·  ethnicity shows the biggest gap; "
             "age is near-identical", fontsize=12, color=INK_2)
    fig.text(0.02, 0.02,
             "Census 2021 via Nomis  ·  Cambridgeshire 43 MSOAs, Birmingham 132  ·  SD = spread across an area's MSOAs",
             fontsize=9, color=MUTED)

    out = IMAGE_DIR / "diversity_summary_table_cambridge_vs_birmingham.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    div, cam, bham = load()
    outs = [make_map(cam, bham, k) for k in DIMS]
    outs.append(make_table(div))
    for o in outs:
        print(f"Wrote {o}")


if __name__ == "__main__":
    main()
