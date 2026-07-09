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


def _fmt_p(pct):
    if pct >= 10:
        return f"{pct:.0f}%"
    if pct >= 1:
        return f"{pct:.1f}%"
    return f"{pct:.2g}%"


def main():
    cam = cambridge_results()
    labels = [lab for _, lab in BLOCKS.values()]  # Ethnicity first

    fig, ax = plt.subplots(figsize=(11.5, 7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.20, right=0.86, top=0.80, bottom=0.12)

    y = np.arange(len(labels))[::-1]  # Ethnicity at top
    y_eth = y[0]

    ax.set_xscale("log")
    # shade the significant zone (p < 5%)
    ax.axvspan(0.08, 5, color="#e6f3ec", zorder=0)
    ax.axvline(5, color="#0e6f4e", lw=1.4, ls="--", zorder=2)
    ax.axhspan(y_eth - 0.5, y_eth + 0.5, color="#f3edfa", zorder=0)  # ethnicity band

    for lab, yy in zip(labels, y):
        pc = cam[lab][1] * 100.0
        pb = BHAM[lab][1] * 100.0
        ax.plot([pc, pb], [yy, yy], color=GRID, lw=2.2, zorder=2)  # connector
        ax.scatter(pc, yy, color=CAM_C, s=150, edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter(pb, yy, color=BHAM_C, s=150, edgecolor="white", linewidth=1.2, zorder=4)
        pts = sorted([(pc, CAM_C), (pb, BHAM_C)], key=lambda t: t[0])
        ax.annotate(_fmt_p(pts[0][0]), (pts[0][0], yy), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=10, fontweight="bold", color=pts[0][1])
        ax.annotate(_fmt_p(pts[1][0]), (pts[1][0], yy), xytext=(9, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, fontweight="bold", color=pts[1][1])

    ax.text(5, len(labels) - 0.35, "significant\n(p < 5%)", ha="right", va="bottom",
            fontsize=9.5, color="#0e6f4e", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")  # Ethnicity
    ax.set_xlim(0.08, 100)
    ax.set_ylim(-0.6, len(labels) - 0.15)
    ax.set_xticks([0.1, 1, 5, 10, 100])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("p-value of the diversity–price correlation  (log scale)", fontsize=11, color=INK_2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)

    # callout on the highlighted ethnicity row
    ax.annotate("Ethnicity: significant in\nBirmingham, not here",
                xy=(BHAM["Ethnicity"][1] * 100.0, y_eth), xytext=(0.16, y_eth + 0.60),
                fontsize=10, color="#5a2d91", fontweight="bold", va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color="#5a2d91", lw=1.2))

    ax.legend(
        handles=[Patch(color=CAM_C, label="Cambridgeshire"),
                 Patch(color=BHAM_C, label="Birmingham")],
        loc="center left", frameon=False, fontsize=10.5, labelcolor=INK_2, bbox_to_anchor=(1.01, 0.5),
    )

    fig.text(0.06, 0.945, "Which Demographics Significantly Track Price?", fontsize=19,
             fontweight="bold", color=INK)
    fig.text(0.06, 0.905,
             "significance (p-value) of each demographic's diversity–price link, per MSOA  ·  "
             "Cambridgeshire vs Birmingham  ·  left of the line = significant", fontsize=11.5, color=INK_2)
    fig.text(0.06, 0.03,
             "p-value of Simpson diversity vs median price  ·  Cambridgeshire computed from the sales data; "
             "Birmingham = team model results", fontsize=9, color=MUTED)

    out = IMAGE_DIR / "demographic_price_correlation_cambridge_vs_birmingham.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    for lab in labels:
        print(f"  {lab:20s} Cam p={cam[lab][1]*100:.2f}%   Bham p={BHAM[lab][1]*100:.2f}%")


if __name__ == "__main__":
    main()
