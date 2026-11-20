"""Plot the primary repeat-scan calibration curves and paired difference."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / "results" / "nki-calibration"
METHODS = [
    ("personal", "Personal baseline", "#006c6b", "o", "-"),
    ("pooled", "Pooled baseline", "#b45428", "s", "--"),
    ("scratch", "Target only", "#514c77", "^", ":"),
]


def main():
    scores = pd.read_csv(OUT / "scores.csv")
    primary = scores[(scores.half == "odd") & (scores.source_visit == 1)]
    coarse = sorted(primary.budget.unique())
    refinement = OUT/"refinement"/"scores.csv"
    if refinement.exists():
        extra = pd.read_csv(refinement)
        primary = pd.concat([primary,extra[~extra.budget.isin(coarse)]], ignore_index=True)
    table = primary.groupby(["subject", "budget", "method"]).r2.mean().unstack("method")
    budgets = sorted(primary.budget.unique())
    x = np.arange(len(budgets))
    measured = table.groupby(level="budget").mean().reindex(budgets)
    paired = (table.personal - table.pooled).unstack("budget").loc[:, budgets]
    assert paired.shape == (25, len(budgets)) and np.isfinite(paired).all().all()
    generator = np.random.default_rng(91001)
    weights = generator.multinomial(25, np.repeat(1/25, 25), size=10_000) / 25
    lo, hi = np.quantile(weights @ paired.to_numpy(), [.025,.975], axis=0)
    diff = paired.mean().to_numpy()

    fig, (curve, contrast) = plt.subplots(2, 1, figsize=(8.6, 6.5), sharex=True,
                                         gridspec_kw={"height_ratios": [2.1, 1]})
    fig.subplots_adjust(left=.13, right=.97, top=.88, bottom=.15, hspace=.35)
    for method, name, color, marker, style in METHODS:
        curve.plot(x, measured[method], color=color, label=name, marker=marker, ls=style,
                   lw=1.8, markersize=5, markerfacecolor="white" if method!="personal" else color)
    curve.axhline(.5, ls=(0,(3,3)), color="#929a9f", lw=1)
    passes = np.logical_and.accumulate((measured.personal.to_numpy()>=.5)[::-1])[::-1]
    if passes.any():
        first = passes.argmax()
        curve.annotate(f"Sustained R² ≥ 0.50\nfrom {budgets[first]} steps",
                       (x[first], measured.personal.iloc[first]),
                       xytext=(x[first]+.8,.29),fontsize=9,color="#014c50",
                       arrowprops={"arrowstyle":"->","color":"#4c7779","lw":.9})
    if measured.min().min() < -2:
        curve.set_yscale("symlog", linthresh=1)
    curve.set_ylabel("Mean held-out R²")
    curve.legend(loc="best", frameon=False, ncol=3, fontsize=9)
    curve.set_title("Accuracy on a new scan", loc="left", fontweight="bold", fontsize=11)
    contrast.axhline(0, color="#687274", lw=1)
    contrast.errorbar(x, diff, yerr=[np.maximum(0, diff-lo), np.maximum(0, hi-diff)],
                      fmt="o", color="#006c6b", ecolor="#498584", capsize=3, markersize=4)
    if np.max(np.abs(np.r_[lo,hi])) > .18:
        contrast.set_yscale("symlog", linthresh=.015)
        contrast.set_ylabel("Difference in R²\n(symmetric log scale)")
    else:
        contrast.set_ylabel("Difference in R²")
    contrast.set_title("Personal minus pooled, with paired 95% intervals", loc="left",
                       fontweight="bold", fontsize=11)
    contrast.set_xticks(x, [f"{int(k):,}" if k in coarse or k in (40,44,48) else "" for k in budgets])
    contrast.set_xlabel("Labeled target time steps")
    for ax in (curve, contrast):
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", lw=.5, alpha=.23)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    fig.suptitle("How much calibration does a new scan need?", x=.13, y=.97, ha="left",
                 fontweight="bold", fontsize=16)
    fig.text(.13, .052, "25 participants, six calibration draws per person. Baseline to repeat scan, odd diffusion directions.",
             fontsize=8.5, color="#364652")
    fig.text(.13, .025, "Horizontal spacing marks successive tested budgets. Intervals resample participants and hold starting decoders fixed.",
             fontsize=8.5, color="#364652")
    fig.savefig(OUT/"calibration_curves.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
