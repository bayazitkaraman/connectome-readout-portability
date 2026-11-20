"""Plot second-scan recall and its paired change when a personal scan joins the donor pool."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"
OWN = "pool_plus_own"
OTHER = "pool_plus_other"
TEAL = "#076d6d"
ORANGE = "#a54c1b"
BLUE = "#485c90"
PRIMARY = "equal_donor"
SENSITIVITY = "fixed_025"


def budget_position(budget):
    # Place the zero-label baseline one major tick before eight on the log2 scale.
    return np.log2(np.maximum(4, budget)) - 2


def draw_cohort(axes, folder, title):
    learning, contrasts = axes
    curves = pd.read_csv(folder / "curves.csv")
    comparisons = pd.read_csv(folder / "contrasts.csv")
    thresholds = pd.read_csv(folder / "thresholds.csv")
    for method, label, color, style, marker in (
        (OWN, "Pool + own scan", TEAL, "-", "o"),
        (OTHER, "Pool + other scan", ORANGE, "--", "s"),
    ):
        part = curves[(curves.mix_rule == PRIMARY) & (curves.method == method)].sort_values("budget")
        learning.plot(budget_position(part.budget), part.mean_r2, label=label, color=color,
                      linestyle=style, marker=marker, markersize=3.7,
                      markerfacecolor="none" if method == OTHER else color, linewidth=1.7)
    learning.axhline(.5, color="#777f88", linewidth=.9, linestyle=":", zorder=0)
    crossing = thresholds[(thresholds.mix_rule == PRIMARY) & (thresholds.threshold == .5) &
                          (thresholds.method_a == OWN) & (thresholds.method_b == OTHER)].iloc[0]
    own_budget, other_budget = crossing[["cohort_a_grid_budget", "cohort_b_grid_budget"]]
    if np.isfinite(own_budget) and own_budget == other_budget:
        learning.annotate(f"Both sustain R² ≥ 0.50\nat {int(own_budget)} examples",
                          (budget_position(own_budget), .5), xytext=(16,-34),
                          textcoords="offset points", fontsize=8, color="#33454c",
                          arrowprops={"arrowstyle":"-", "color":"#63737b", "lw":.8})
    ticks = [0, 8, 16, 32, 64, 128, 256, 512, 2352]
    learning.set_xticks(budget_position(ticks), [f"{b:,}" for b in ticks], fontsize=8)
    learning.set_ylim(-.06, 1)
    learning.set_ylabel("Mean held-out R²")
    learning.set_title(f"{title}  ·  Recall", fontsize=11, fontweight="bold", loc="left")

    selected = comparisons[(comparisons.method_a == OWN) & (comparisons.method_b == OTHER)]
    early = [0, 8, 16, 32, 64]
    for rule, label, color, offset, marker in (
        (PRIMARY, "Equal share per donor", TEAL, -.065, "o"),
        (SENSITIVITY, "25% from added scan", BLUE, .065, "D"),
    ):
        part = selected[selected.mix_rule == rule].set_index("budget").loc[early]
        estimate = part.r2_difference.to_numpy()
        uncertainty = np.vstack([estimate - part.r2_ci_low, part.r2_ci_high - estimate])
        contrasts.errorbar(np.arange(len(early)) + offset, estimate, yerr=uncertainty, fmt=marker,
                           label=label, color=color, markersize=4, capsize=3, linewidth=1.1)
    contrasts.set_xticks(np.arange(len(early)), early)
    contrasts.set_xlim(-.35, len(early)-.65)
    contrasts.axhline(0, color="#737b83", linewidth=.9, zorder=0)
    contrasts.set_ylabel("Own minus other (R²)")
    contrasts.set_title(f"{title}  ·  Paired accuracy difference", fontsize=11, fontweight="bold", loc="left")
    for ax in axes:
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y", linewidth=.45, alpha=.24)
        ax.set_axisbelow(True)


def plot():
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.9), sharex="col", sharey="col")
    for row, (cohort, name) in enumerate((("nki","NKI–Rockland, 25 people"), ("bnu1","Beijing, 57 people"))):
        draw_cohort(axes[row], RESULTS / f"{cohort}-augmentation", name)
    for ax in axes[1]:
        ax.set_xlabel("Labeled examples from the second scan")
    fig.subplots_adjust(left=.085, right=.975, bottom=.13, top=.80, wspace=.28, hspace=.32)
    fig.suptitle("Whose first scan is added to the pooled decoder?", x=.085, y=.975,
                 ha="left", fontsize=17, fontweight="bold")
    for ax, x in zip(axes[0], [.075,.565]):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x,.918), ncol=2,
                   frameon=False, fontsize=8.4, handlelength=1.8, columnspacing=1.5)
    fig.text(.085,.037,"Left panels use equal contributions per donor. Right panels show paired 95% participant-bootstrap intervals.",
             fontsize=8.5, color="#384b52")
    destination = RESULTS / "pool-augmentation"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / "calibration.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot()
