"""Does demographic diversity track house price? Cambridgeshire vs Birmingham.

Grouped bars comparing, for each demographic, the correlation (Pearson r) between
its Simpson diversity per MSOA and area house prices. Cambridgeshire is computed
from the cleaned sales data; Birmingham's values are the team's Birmingham model
results. The ethnicity row is highlighted — it flips sign between the two cities.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cambridgeshire_maps import (  # noqa: E402
    PROJECT_ROOT, IMAGE_DIR, COL_RENAME,
    SURFACE, INK, INK_2, MUTED, GRID,
)

CLEANED = PROJECT_ROOT / "data" / "cleaned" / "Cambridge data_cleaned.csv"

# block -> census columns (readable names after COL_RENAME) and display label
BLOCKS = {
    "Ethnicity": (["Asian", "Black", "Mixed", "White", "Other"], "Ethnicity"),
    "Age": (["Minors 0-18", "Adults 18-60", "Elders >60"], "Age"),
    "Commute distance": (["Work from Home (0km)", "0-5km", "5-30km", ">30km", "Living Offshore"], "Distance to Work"),
    "Education": (["Low", "Medium", "High", "Other Qualification"], "Level of Education"),
    "Commute method": (["City Public", "Rail", "Cycle", "Driving", "Foot", "Other Method"], "Type of Commute"),
}

# team's Birmingham model results: display label -> (r, p)
BHAM = {
    "Ethnicity": (-0.250260, 0.0038),
    "Age": (0.272605, 0.0015),
    "Distance to Work": (-0.106990, 0.2221),
    "Level of Education": (0.143418, 0.101),
    "Type of Commute": (-0.238979, 0.005786),
}

CAM_C, BHAM_C = "#2a78d6", "#7a3fb0"


def cambridge_results():
    df = pd.read_csv(CLEANED).rename(columns=COL_RENAME)
    per = df.groupby("msoa21").first()
    per["mp"] = df.groupby("msoa21")["price_sold"].median()
    out = {}
    for cols, label in BLOCKS.values():
        p = per[cols].div(per[cols].sum(axis=1), axis=0)
        k = p.shape[1]
        simp = (1 - (p ** 2).sum(axis=1)) / (1 - 1 / k)
        r, pv = pearsonr(simp, per["mp"])
        out[label] = (float(r), float(pv))
    return out


def main():
    cam = cambridge_results()
    labels = [lab for _, lab in BLOCKS.values()]  # Ethnicity first

    fig, ax = plt.subplots(figsize=(11.5, 7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.20, right=0.78, top=0.80, bottom=0.11)

    y = np.arange(len(labels))[::-1]  # Ethnicity at top
    h = 0.36

    # highlight band behind the ethnicity row
    y_eth = y[0]
    ax.axhspan(y_eth - 0.5, y_eth + 0.5, color="#f3edfa", zorder=0)

    def bar(row_y, val, color):
        ax.barh(row_y, val, height=h, color=color, zorder=3)

    for lab, yy in zip(labels, y):
        rc, pc = cam[lab]
        rb, pb = BHAM[lab]
        bar(yy + h / 2 + 0.02, rc, CAM_C)
        bar(yy - h / 2 - 0.02, rb, BHAM_C)
        for val, pv, ry in ((rc, pc, yy + h / 2 + 0.02), (rb, pb, yy - h / 2 - 0.02)):
            star = "*" if pv < 0.05 else ""
            ax.text(val + (0.012 if val >= 0 else -0.012), ry, f"{val:+.2f}{star}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=10.5, fontweight="bold", color=INK_2)

    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")  # Ethnicity
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Correlation of demographic diversity with house price  (Pearson r)",
                  fontsize=11, color=INK_2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.6, zorder=0)

    # callout on the highlighted ethnicity row
    ax.annotate("Ethnicity flips sign:\n+0.15 (n.s.) here vs −0.25 in Birmingham",
                xy=(BHAM["Ethnicity"][0], y_eth - h / 2 - 0.02), xytext=(-0.47, y_eth + 0.62),
                fontsize=10, color="#5a2d91", fontweight="bold", va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color="#5a2d91", lw=1.2))

    leg = ax.legend(
        handles=[Patch(color=CAM_C, label="Cambridgeshire  (43 MSOAs)"),
                 Patch(color=BHAM_C, label="Birmingham  (132 MSOAs)")],
        loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK_2,
        bbox_to_anchor=(1.0, 0.0),
    )
    ax.text(0.995, 0.16, "* significant at p < 0.05", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color=MUTED)

    fig.text(0.06, 0.945, "Does Demographic Diversity Track House Price?", fontsize=19,
             fontweight="bold", color=INK)
    fig.text(0.06, 0.905,
             "correlation of each demographic's diversity with area prices, per MSOA  ·  "
             "Cambridgeshire vs Birmingham  ·  ethnicity is the sharpest contrast", fontsize=11.5, color=INK_2)
    fig.text(0.06, 0.03,
             "Pearson r of Simpson diversity vs median price  ·  Cambridgeshire computed from the sales data; "
             "Birmingham = team model results", fontsize=9, color=MUTED)

    out = IMAGE_DIR / "demographic_price_correlation_cambridge_vs_birmingham.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    for lab in labels:
        print(f"  {lab:20s} Cam r={cam[lab][0]:+.3f} (p={cam[lab][1]*100:.1f}%)   Bham r={BHAM[lab][0]:+.3f}")


if __name__ == "__main__":
    main()
