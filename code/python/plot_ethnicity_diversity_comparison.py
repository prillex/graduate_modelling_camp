"""Ethnicity diversity: Cambridgeshire vs Birmingham, side by side.

Gini-Simpson diversity of the five high-level Census 2021 ethnic groups
(Asian, Black, Mixed, White, Other) per MSOA, mapped for the project's 43
Cambridgeshire MSOAs next to Birmingham's 132 MSOAs on one shared colour scale.

Ethnic-group counts are Census 2021 table TS021 (ONS / Nomis), stored in
data/spatial/ethnicity_ts021_cam_bham.csv so the figure is reproducible offline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cambridgeshire_maps import (  # noqa: E402
    PROJECT_ROOT, IMAGE_DIR, BOUNDARIES,
    SURFACE, INK, INK_2, MUTED, VIOLET_RAMP,
)

ETH_CSV = PROJECT_ROOT / "data" / "spatial" / "ethnicity_ts021_cam_bham.csv"
CLEANED = PROJECT_ROOT / "data" / "cleaned" / "Cambridge data_cleaned.csv"
ETH_COLS = ["Asian", "Black", "Mixed", "White", "Other"]


def gini_simpson(counts):
    p = counts.div(counts.sum(axis=1), axis=0)
    return 1.0 - (p ** 2).sum(axis=1)


def main():
    eth = pd.read_csv(ETH_CSV).set_index("msoa21")
    eth["diversity"] = gini_simpson(eth[ETH_COLS])

    cam_codes = sorted(pd.read_csv(CLEANED, usecols=["msoa21"])["msoa21"].dropna().unique())
    gdf = gpd.read_file(BOUNDARIES, columns=["MSOA21CD", "MSOA21NM", "geometry"])
    gdf = gdf[gdf["MSOA21CD"].isin(eth.index)].to_crs(27700)
    gdf = gdf.merge(eth["diversity"], left_on="MSOA21CD", right_index=True)

    cam = gdf[gdf["MSOA21CD"].isin(cam_codes)]
    bham = gdf[gdf["MSOA21NM"].str.startswith("Birmingham", na=False)]

    norm = Normalize(gdf["diversity"].min(), gdf["diversity"].max())

    fig = plt.figure(figsize=(14, 8.6))
    fig.patch.set_facecolor(SURFACE)
    ax_c = fig.add_axes([0.02, 0.16, 0.46, 0.70])
    ax_b = fig.add_axes([0.50, 0.16, 0.46, 0.70])

    for ax, region, name in ((ax_c, cam, "Cambridgeshire"), (ax_b, bham, "Birmingham")):
        ax.set_facecolor(SURFACE)
        region.plot(ax=ax, column="diversity", cmap=VIOLET_RAMP, norm=norm,
                    linewidth=0.4, edgecolor="white")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=17, fontweight="bold", color=INK, pad=6)
        ax.text(0.5, -0.02, f"{len(region)} MSOAs   ·   avg diversity {region['diversity'].mean():.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=11.5, color=INK_2)

    # shared horizontal colorbar
    cax = fig.add_axes([0.30, 0.085, 0.40, 0.02])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=VIOLET_RAMP, norm=norm), cax=cax,
                      orientation="horizontal")
    cb.set_label("Ethnic diversity  —  Gini-Simpson index  (chance two random residents differ)",
                 color=INK_2, fontsize=10.5)
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)

    fig.text(0.02, 0.955, "Ethnic Diversity: Cambridgeshire vs Birmingham", fontsize=21,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.915,
             "Gini-Simpson diversity of the five Census 2021 ethnic groups per MSOA, one shared scale  ·  "
             "darker = more mixed  ·  Birmingham is far more diverse", fontsize=12.5, color=INK_2)
    fig.text(0.02, 0.03,
             "Source: ONS Census 2021, table TS021 (Ethnic group), via Nomis  ·  Boundaries: ONS MSOA 2021  ·  EPSG:27700",
             fontsize=9, color=MUTED)

    out = IMAGE_DIR / "ethnicity_diversity_cambridge_vs_birmingham.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Cambridgeshire avg {cam['diversity'].mean():.3f}  ·  Birmingham avg {bham['diversity'].mean():.3f}")


if __name__ == "__main__":
    main()
