"""Plot Figures 2 and 3 using only archived readout scores and intervals."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COHORTS = {"nki": ("NKI", 25, 10.0), "bnu1": ("Beijing", 57, 1.0)}
COHORT_STYLES = {"nki": ("o", "#006a91"), "bnu1": ("^", "#b64d11")}
CONDITIONS = {
    "source": ("Source", "#28343b", "o", (0, (3, 2))),
    "numerical_perturbation": ("Numerical rerun", "#368064", "s", (0, (4, 2))),
    "split_half": ("Other direction half", "#765299", "^", (0, (1.2, 1.7))),
    "between_visit": ("Own repeat", "#006a91", "D", "-"),
}


def axis_style(ax):
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#737b80")
    ax.tick_params(length=3, width=.6)
    ax.set_axisbelow(True)


def panel(ax, letter, title):
    ax.set_title(r"$\bf{(" + letter + ")}$  " + title, loc="left", pad=6, fontsize=9.5)


def export(fig, out, basename):
    fig.savefig(out / (basename + ".pdf"), facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / (basename + ".png"), dpi=600, facecolor="white")
    plt.close(fig)


def primary_paths(results):
    path = results / "portability-path"
    visits = pd.read_csv(path / "visit_people.csv.gz", dtype={"target_id": str})
    visits = visits[visits.task.eq("linear") & visits.rung.eq("between_visit")]
    means = visits.groupby(["cohort", "alpha"])[["source_r2", "target_r2"]].mean()
    cohort_paths = {}
    summaries = pd.read_csv(path / "path_summary.csv")
    for cohort, (_, n, _) in COHORTS.items():
        rows = visits[visits.cohort.eq(cohort)]
        assert rows.groupby("alpha").target_id.nunique().eq(n).all()
        pair = means.loc[cohort].sort_index()
        checks = summaries[summaries.cohort.eq(cohort) & summaries.task.eq("linear")
                           & summaries.rung.eq("between_visit")].set_index("alpha").sort_index()
        np.testing.assert_allclose(pair, checks[["source_r2", "target_r2"]], atol=1e-10)
        cohort_paths[cohort] = pair.rename(columns={"source_r2": "source", "target_r2": "between_visit"})

    technical = pd.read_csv(path / "rung_pairs.csv.gz", dtype={"source_id": str})
    technical = technical[technical.task.eq("linear") & technical.source_visit.eq(1)
                          & technical.source_half.eq("odd") & technical.rung.isin(["split_half", "numerical_perturbation"])]
    groups = technical.groupby(["alpha", "rung", "source_id"])
    counts = groups.size()
    assert counts.xs("split_half", level="rung").eq(1).all()
    assert counts.xs("numerical_perturbation", level="rung").eq(20).all()
    participants = groups.target_r2.mean()
    assert participants.groupby(["alpha", "rung"]).size().eq(25).all()
    technical_means = participants.groupby(["alpha", "rung"]).mean().unstack("rung")
    cohort_paths["nki"] = cohort_paths["nki"].join(technical_means, validate="one_to_one")

    matched = pd.read_csv(path / "shift_summary.csv")
    matched = matched[matched.source_set.eq("first_visit") & matched.task.eq("linear")]
    for row in matched.itertuples():
        if row.rung in {"numerical_perturbation", "split_half", "between_visit"}:
            value = cohort_paths[row.cohort].loc[row.alpha, row.rung]
            np.testing.assert_allclose(value, row.target_r2, atol=1e-10)
    return cohort_paths


def penalty_figure(results, out):
    paths = primary_paths(results)
    fig, axes = plt.subplots(2, 2, figsize=(6.70, 5.40))
    fig.subplots_adjust(left=.124, right=.978, top=.929, bottom=.166, wspace=.36, hspace=.46)
    for col, (cohort, (name, n, selected)) in enumerate(COHORTS.items()):
        group = paths[cohort]
        group = group.loc[(group.index >= 1e-14) & (group.index <= 1e4)]
        log_alpha = np.log10(group.index.to_numpy())
        np.testing.assert_allclose(log_alpha, np.arange(-14, 5), atol=1e-12)
        for row in [0, 1]:
            ax = axes[row, col]
            axis_style(ax)
            ax.axhline(0, color="#a3afb5", linewidth=.6, zorder=0)
            for key, (_, color, marker, line) in CONDITIONS.items():
                if key not in group:
                    continue
                ax.plot(log_alpha, group[key], color=color, linestyle=line, linewidth=1.15,
                        marker=marker, markersize=2.9, mew=.2, mec="white", zorder=2)
            ax.scatter(np.log10(selected), group.loc[selected, "between_visit"],
                       s=41, zorder=4, facecolors="none", edgecolors="#162229", linewidths=.85)
            if row == 0:
                ax.set_yscale("symlog", linthresh=1)
                ax.set(xlim=(-14.4, 4.4), ylim=(-7e11, 1.15))
                ax.set_xticks([-14, -10, -6, -2, 2, 4])
                ax.set_yticks([-1e11, -1e8, -1e5, -1e2, 0, 1],
                              [r"$-10^{11}$", r"$-10^{8}$", r"$-10^{5}$", r"$-10^{2}$", "0", "1"])
                ax.yaxis.set_minor_locator(NullLocator())
                panel(ax, "ab"[col], f"{name} ({n}), full grid")
                if col == 0:
                    ax.set_ylabel("Mean R² (symmetric log)")
            else:
                ax.set(xlim=(-3.18, 3.12), ylim=(-.215, .65), xlabel="log₁₀(relative ridge penalty)")
                ax.xaxis.set_major_locator(FixedLocator(np.arange(-3, 4)))
                ax.set_yticks([-.2, 0, .2, .4, .6], ["−0.2", "0", "0.2", "0.4", "0.6"])
                panel(ax, "cd"[col], f"{name}, detail")
                if col == 0:
                    ax.set_ylabel("Mean R² (linear scale)")
    handles = [Line2D([], [], label=label, color=color, marker=marker, linestyle=line,
                      markersize=4, linewidth=1.1) for label, color, marker, line in CONDITIONS.values()]
    handles.append(Line2D([], [], label="Transfer-selected repeat", color="#162229", marker="o",
                          markersize=6, markerfacecolor="none", markeredgewidth=.9, linestyle="none"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .008), frameon=False,
               ncol=3, fontsize=8, handlelength=1.65, columnspacing=1.2)
    export(fig, out, "Figure_2")


def alignment_figure(results, out):
    data = results / "portability-baseline"
    people = pd.read_csv(data / "people.csv", dtype={"target_id": str})
    summaries = pd.read_csv(data / "summary.csv").set_index("cohort")
    low, high = -.405, .305
    fig = plt.figure(figsize=(6.70, 5.35))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.85, 1], left=.145, right=.975,
                            top=.927, bottom=.109, wspace=.34, hspace=.55)
    gains_ax = fig.add_subplot(grid[1, :])
    axis_style(gains_ax)
    gains_ax.axvline(0, ls=(0, (2, 2)), color="#677279", lw=.85, zorder=0)
    for i, (cohort, (name, n, penalty)) in enumerate(COHORTS.items()):
        scores = people[people.cohort.eq(cohort)].sort_values("target_id")
        summary = summaries.loc[cohort]
        assert scores.target_id.nunique() == len(scores) == summary.n_people == n
        assert scores.alpha.eq(penalty).all() and scores.unlabeled_target_rows.eq(2352).all()
        differences = (scores.aligned_r2 - scores.untouched_r2).to_numpy()
        np.testing.assert_allclose(differences, scores.delta_r2, atol=1e-13)
        assert np.sum(differences > 0) == summary.n_improved
        rng = np.random.default_rng(20261120)
        draws = rng.integers(n, size=(10000, n))
        interval = np.quantile(differences[draws].mean(axis=1), [.025, .975])
        np.testing.assert_allclose(interval, [summary.delta_lo, summary.delta_hi], atol=1e-13)
        np.testing.assert_allclose(differences.mean(), summary.delta_r2, atol=1e-13)

        marker, color = COHORT_STYLES[cohort]
        ax = fig.add_subplot(grid[0, i])
        axis_style(ax)
        panel(ax, "ab"[i], f"{name} ({n})")
        zero_style = {"color": "#b3bcc0", "lw": .55, "zorder": 0}
        ax.axhline(0, **zero_style)
        ax.axvline(0, **zero_style)
        ax.plot([low, high], [low, high], color="#526068", ls=(0, (4, 3)), linewidth=.75, zorder=1)
        for col in ["untouched_r2", "aligned_r2"]:
            assert scores[col].between(low, high).all()
        ax.scatter(scores.untouched_r2, scores.aligned_r2, s=16, marker=marker,
                   color=color, alpha=.71, edgecolors="white", linewidths=.3, zorder=2)
        ax.set(xlim=(low, high), ylim=(low, high), aspect="equal")
        ax.set_xticks([-.4, -.2, 0, .2], ["−0.4", "−0.2", "0", "0.2"])
        ax.set_yticks([-.4, -.2, 0, .2], ["−0.4", "−0.2", "0", "0.2"])
        ax.set_xlabel("Untouched repeat R²")
        if i == 0:
            ax.set_ylabel("Aligned repeat R²")

        jitter = np.random.default_rng(570+i).uniform(-.11, .11, size=n)
        gains_ax.scatter(differences, i-.07+jitter, color=color, alpha=.59, s=13,
                         marker=marker, edgecolors="white", linewidths=.22)
        gains_ax.errorbar(summary.delta_r2, i+.20,
                          xerr=[[summary.delta_r2-summary.delta_lo], [summary.delta_hi-summary.delta_r2]],
                          color="#192930", ecolor="#192930", fmt="D", linewidth=1.5,
                          capsize=2.7, markersize=4.1, zorder=4)
    panel(gains_ax, "c", "Within-participant gain")
    gains_ax.set(xlim=(-.18, .33), ylim=(1.45, -.42),
                 xlabel="Aligned minus untouched repeat R²")
    gains_ax.set_yticks([0, 1], ["NKI", "Beijing"])
    gains_ax.set_xticks([-.15, -.10, -.05, 0, .05, .10, .15, .20, .25, .30],
                        ["−0.15", "−0.10", "−0.05", "0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"])
    assert people.delta_r2.between(*gains_ax.get_xlim()).all()
    export(fig, out, "Figure_3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "portability-path")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "axes.labelsize": 8.2,
                         "axes.linewidth": .7, "axes.labelcolor": "#243038",
                         "xtick.color": "#243038", "ytick.color": "#243038",
                         "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42})
    penalty_figure(args.results, args.out)
    alignment_figure(args.results, args.out)


if __name__ == "__main__":
    main()
