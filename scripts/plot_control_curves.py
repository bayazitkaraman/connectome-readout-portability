"""Plot measured and spatially constrained surrogate recall on held-out networks."""
import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"
CONTROL_RESULTS = RESULTS / "control-comparison"
DELAYS = np.arange(1, 11)
METRICS = ["shared_r2", "own_r2"]


def load_curves(family):
    real = pd.read_csv(RESULTS / "sensitivity_participant_scores.csv")
    real = real.loc[(real.map_key == 10) & np.isclose(real.gain, .9)
                    & real.decoder.isin(["shared", "individual"])]
    observed = real.pivot(index=["fold", "participant", "delay"], columns="decoder",
                          values="r2").rename(columns={"individual": "own_r2", "shared": "shared_r2"})
    control = pd.read_csv(CONTROL_RESULTS / f"{family}_scores.csv")
    if set(control.family) != {family}:
        raise ValueError(f"Expected only records from {family}.")
    participants = observed.index.get_level_values("participant").nunique()
    draws = control.draw.nunique()
    if participants != 70 or draws != 24 or len(observed) != participants * len(DELAYS):
        raise ValueError("Expected complete scores for 70 people, 10 delays and 24 surrogate draws.")
    if not np.isfinite(observed[METRICS].to_numpy()).all():
        raise ValueError("Observed scores contain nonfinite values.")
    if not np.isfinite(control[METRICS].to_numpy()).all():
        raise ValueError("Control scores contain nonfinite values.")
    for _, group in control.groupby("draw"):
        pairs = group.set_index(["fold", "participant", "delay"]).index.sort_values()
        if not pairs.equals(observed.index.sort_values()):
            raise ValueError("Surrogate folds, participants or delays differ from observed scores.")
    measured = observed.groupby(level="delay")[METRICS].mean()
    surrogate = control.groupby(["draw", "delay"])[METRICS].mean().groupby(level="delay").mean()
    if not (np.array_equal(measured.index, DELAYS) and np.array_equal(surrogate.index, DELAYS)):
        raise ValueError("A requested delay is missing.")
    return measured, surrogate, participants, draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("spatial", "spatial_budgeted"), default="spatial")
    family = parser.parse_args().family
    measured, surrogate, participants, draws = load_curves(family)
    label = "Spatial surrogate" if family == "spatial" else "Rewired with regional strength constraint"
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9.6, 4.65))
    fig.subplots_adjust(left=.095, right=.977, bottom=.25, top=.705, wspace=.14)
    fig.text(.095, .932, "Memory recall on measured and rewired connectomes",
             fontsize=17, fontweight="bold", color="#20333a")
    lines = []
    for ax, metric, heading in zip(axes, METRICS, ["Shared readout", "Individual readout"]):
        m, = ax.plot(DELAYS, measured[metric], color="#175e69", linewidth=2.2, marker="o",
                     markersize=4.8, label="Measured Lausanne wiring", zorder=3)
        s, = ax.plot(DELAYS, surrogate[metric], color="#a35327", linewidth=1.9,
                     marker="s", markersize=4.1, markerfacecolor="white",
                     markeredgewidth=1.3, linestyle=(0, (4, 2)), label=label)
        lines = [m, s]
        ax.set_title(heading, loc="left", pad=10, fontsize=11.5, fontweight="bold", color="#20333a")
        ax.grid(axis="y", color="#d6ddde", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(.68, 10.32)
        ax.set_xticks(DELAYS)
        ax.tick_params(length=3.0, color="#73868a", labelcolor="#283b40", labelsize=9)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#93a1a2")
        ax.set_xlabel("Recall delay (simulation steps)", labelpad=7, fontsize=10)
    lo = min(0.0, measured.to_numpy().min(), surrogate.to_numpy().min())
    start = np.floor(lo * 4) / 4
    axes[0].set_ylim(start - .025, 1.045)
    axes[0].yaxis.set_major_locator(FixedLocator(np.arange(start, 1.01, .25)))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].set_ylabel("Mean held-out R²", fontsize=10.5, labelpad=7)
    fig.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left",
               bbox_to_anchor=(.09, .887), ncol=2, columnspacing=2.3, borderaxespad=0,
               handlelength=2.75, fontsize=10)
    fig.text(.095, .084, f"Each point averages held-out scores for {participants} participants. "
             f"Surrogate scores also average {draws} rewired cohorts.",
             fontsize=9.1, color="#42595e")
    fig.text(.095, .04, "Both conditions use identical input positions, participant folds and test signals. "
             "Spectral radius 0.90.", fontsize=9.1, color="#42595e")
    filename = "delay_curves.png" if family == "spatial" else "strength_matched_delay_curves.png"
    fig.savefig(CONTROL_RESULTS / filename, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
