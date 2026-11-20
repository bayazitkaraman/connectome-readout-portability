"""Render the paper's three-panel Figure 1 from the fixed analysis tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/portability-path"
OUT = DATA
STYLES = {"nki": ("NKI", "o", "#006a91"), "bnu1": ("Beijing", "^", "#b64d11")}
DARK = "#1a2830"
RULE = "#737b80"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "text.color": DARK, "axes.labelcolor": DARK,
                     "xtick.color": DARK, "ytick.color": DARK,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
                     "axes.linewidth": 0.65})


def style_axes(ax):
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(RULE)
    ax.tick_params(length=3, width=.6, pad=3)
    ax.set_axisbelow(True)


def panel_title(ax, label, title):
    ax.text(0, 1.055, f"({label})", transform=ax.transAxes, fontsize=10,
            weight="bold", va="baseline")
    ax.text(.058 if label=="a" else .12, 1.055, title, transform=ax.transAxes,
            fontsize=9.5, va="baseline", weight="medium")


def main():
    people = pd.read_csv(DATA/"shift_people.csv", dtype={"source_id": str})
    summaries = pd.read_csv(DATA/"shift_summary.csv")
    keep = lambda x: x[x.task.eq("linear") & x.source_set.eq("first_visit")]
    people, summaries = keep(people), keep(summaries)
    weak = people[people.alpha.eq(.001)]
    fig = plt.figure(figsize=(6.70, 5.60), facecolor="white")
    # Explicit positions keep axes and labels independent of backend text metrics.
    a = fig.add_axes([.242, .675, .722, .24])
    b = fig.add_axes([.119, .164, .345, .347])
    c = fig.add_axes([.615, .164, .345, .347])
    for ax in (a,b,c):
        style_axes(ax)

    conditions = [("nki", "numerical_perturbation", "NKI numerical rerun"),
                  ("nki", "split_half", "NKI other direction half"),
                  ("nki", "between_visit", "NKI repeat visit"),
                  ("bnu1", "between_visit", "Beijing repeat visit")]
    for row, (cohort, rung, name) in enumerate(conditions):
        block = weak[weak.cohort.eq(cohort) & weak.rung.eq(rung)].sort_values("source_id")
        assert len(block) == (25 if cohort == "nki" else 57)
        summary = summaries[summaries.cohort.eq(cohort) & summaries.rung.eq(rung)
                            & summaries.alpha.eq(.001)].iloc[0]
        np.testing.assert_allclose(summary.shift_pct_median, block.shift_pct.median(), atol=1e-12)
        jitter = np.random.default_rng(417+row).uniform(-.19, .19, len(block))
        _, marker, color = STYLES[cohort]
        a.scatter(block.shift_pct, row+jitter, s=13.5, marker=marker, facecolor=color,
                  edgecolors="none", alpha=.62, zorder=2)
        a.scatter(summary.shift_pct_median, row, s=32, marker="D", facecolor=DARK,
                  edgecolors="white", linewidths=.65, zorder=3)
    a.set_xscale("log")
    a.set(xlim=(.9, 310), ylim=(3.50,-.5), xlabel="Displacement (% of source training RMS, log scale)")
    a.set_xticks([1,3,10,30,100,300], ["1","3","10","30","100","300"])
    a.xaxis.set_minor_locator(NullLocator())
    a.set_yticks(range(4), [label for _,_,label in conditions])
    a.tick_params(axis="y", length=0, pad=6)
    a.spines["left"].set_visible(False)
    a.grid(axis="x", color="#d1d7da", lw=.5, alpha=.85)
    panel_title(a, "a", "State displacement")

    for ax, low_penalty in [(b,True), (c,False)]:
        if low_penalty:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("symlog", linthresh=1)
            ticks, labs = [-100,-10,-1,0], ["−100","−10","−1","0"]
            ax.set_xticks(ticks,labs)
            ax.set_yticks(ticks,labs)
            ax.set_xlim(-435, .40)
            ax.set_ylim(-435, .40)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            lower, upper = -435, .40
        else:
            ax.set_xlim(.095,.285)
            ax.set_xticks([.10,.15,.20,.25], ["0.10","0.15","0.20","0.25"])
            ax.set_ylim(-.40,.29)
            ax.set_yticks([-.4,-.2,0,.2], ["−0.4","−0.2","0","0.2"])
            lower, upper = ax.get_xlim()
        ax.plot([lower,upper], [lower,upper], ls=(0,(4,3)), color=DARK, lw=.80, zorder=1)
        ax.axhline(0, color="#adb6ba", lw=.55, zorder=0)
        if low_penalty:
            ax.axvline(0, color="#adb6ba", lw=.55, zorder=0)
        ax.set_xlabel("Expected R² under matched noise")
        ax.set_ylabel("Measured repeat-scan R²")
        for cohort, (_, marker, color) in STYLES.items():
            alpha = .001 if low_penalty else {"nki":10., "bnu1":1.}[cohort]
            points = people[people.cohort.eq(cohort) & people.rung.eq("between_visit")
                            & people.alpha.eq(alpha)]
            assert len(points) == (25 if cohort == "nki" else 57)
            assert points.isotropic_r2.between(*ax.get_xlim()).all()
            assert points.target_r2.between(*ax.get_ylim()).all()
            ax.scatter(points.isotropic_r2, points.target_r2, s=16, marker=marker,
                       facecolor=color, edgecolor="white", linewidths=.26, alpha=.83,
                       zorder=(3 if cohort == "nki" else 2))
    panel_title(b, "b", "Weak penalty (α = 0.001)")
    panel_title(c, "c", "Transfer-selected penalty")
    b.text(.975, .055, "Symmetric log axes", transform=b.transAxes, fontsize=8,
           va="bottom", ha="right", color="#47565c")
    handles = [Line2D([],[],marker=m, color="none", markeredgecolor="none", markerfacecolor=col,
                      markersize=5, label=f"{name} ({n})")
               for (name,m,col), n in zip(STYLES.values(), [25,57])]
    handles += [Line2D([],[],marker="D", color="none", markeredgecolor="none",
                       markerfacecolor=DARK, markersize=4.6, label="Median in (a)"),
                Line2D([],[],color=DARK, linewidth=.8, ls=(0,(4,3)), label="Equal scores")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.012), frameon=False,
               fontsize=8, ncol=4, columnspacing=1.55, handlelength=1.5, handletextpad=.4)
    OUT.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUT/"Figure_1.pdf", metadata={"CreationDate":None,"ModDate":None}, facecolor="white")
    fig.savefig(OUT/"Figure_1.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
