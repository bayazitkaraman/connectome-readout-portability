"""Describe participant distributions and single-target influence on frozen transfer curves."""
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from summarize_portability_path import KEY, OUT, PATH, SIZES, STYLE


def participant_scores():
    visits = pd.read_csv(OUT / "visit_people.csv.gz", dtype={"target_id": str})
    technical = pd.read_csv(OUT / "rung_pairs.csv.gz", dtype={"target_id": str})
    numerical = technical.loc[technical.rung.eq("numerical_perturbation")].copy()
    changed = numerical.loc[numerical.identical_offdiag.eq(0)].assign(rung="numerical_changed_only")
    by_person = KEY + ["target_id"]
    cells = pd.concat([technical, changed], ignore_index=True).groupby(
        by_person + ["source_visit", "source_half"]
    ).target_r2.mean()
    # Equalize the four original scan halves even when changed rerun counts differ.
    if not cells.groupby(by_person).size().eq(4).all():
        raise ValueError("Incomplete participant visit/half coverage")
    people = pd.concat([visits[by_person + ["target_r2"]],
                        cells.groupby(by_person).mean().reset_index()], ignore_index=True)
    if people.duplicated(by_person).any() or not np.isfinite(people.target_r2).all():
        raise ValueError("Incomplete or duplicate target-person path")
    return people, numerical


def summarize(people):
    rows = []
    for (cohort, task, rung), group in people.groupby(KEY[:-1]):
        paths = group.pivot(index="target_id", columns="alpha", values="target_r2")
        n = len(paths)
        np.testing.assert_allclose(paths.columns, PATH, atol=0, rtol=1e-12)
        if n != SIZES[cohort] or paths.isna().any().any():
            raise ValueError("Participant path is incomplete")
        scores = paths.to_numpy()
        mean = scores.mean(axis=0)
        # Omit one evaluation target at a time. Fitted readouts and donor bank stay fixed.
        leave_one_out = np.array([np.delete(scores, i, axis=0).mean(axis=0) for i in range(n)])
        peak_counts = np.bincount(leave_one_out.argmax(axis=1), minlength=len(PATH))
        leave_two_out = np.array([np.delete(scores, pair, axis=0).mean(axis=0)
                                  for pair in combinations(range(n), 2)])
        pair_peak_counts = np.bincount(leave_two_out.argmax(axis=1), minlength=len(PATH))
        medians = np.median(scores, axis=0)
        q10, q90 = np.quantile(scores, [.1, .9], axis=0)
        positives = (scores > 0).sum(axis=0)
        for j, alpha in enumerate(paths.columns):
            rows.append(dict(cohort=cohort, task=task, rung=rung, alpha=alpha, n_people=n,
                             target_r2_mean=mean[j], target_r2_median=medians[j],
                             target_r2_p10=q10[j], target_r2_p90=q90[j],
                             n_people_positive=int(positives[j]), fraction_people_positive=positives[j] / n,
                             mean_r2_omit_one_min=leave_one_out[:, j].min(),
                             mean_r2_omit_one_max=leave_one_out[:, j].max(),
                             peak_of_full_mean=int(j == mean.argmax()),
                             n_omit_one_means_peaking_here=int(peak_counts[j]),
                             n_omit_two_means_peaking_here=int(pair_peak_counts[j]),
                             n_omit_two_combinations=len(leave_two_out)))
    result = pd.DataFrame(rows)
    previous = pd.read_csv(OUT / "path_summary.csv")
    matched = result.merge(previous, on=KEY, validate="one_to_one")
    if len(matched) != len(result) or len(matched) != len(previous):
        raise ValueError("Robust path does not cover the archived cells")
    np.testing.assert_allclose(matched.target_r2_mean, matched.target_r2, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(matched.target_r2_median_x, matched.target_r2_median_y,
                               rtol=1e-11, atol=1e-14)
    return result


def summarize_matrices(numerical):
    # Rows here are matrices, unlike the participant-level estimand above.
    records = []
    for (task, alpha, identical), group in numerical.groupby(["task", "alpha", "identical_offdiag"]):
        values = group.target_r2.to_numpy()
        records.append(dict(task=task, alpha=alpha, matrix_status="identical" if identical else "changed",
                            n_people=group.target_id.nunique(), n_matrices=len(group),
                            median_matrix_r2=np.median(values), n_matrices_positive=int((values > 0).sum()),
                            fraction_matrices_positive=(values > 0).mean()))
    return pd.DataFrame(records)


def plot(result):
    styles = {**STYLE, "numerical_changed_only": ("Numerical, changed only", "#5b5935", "x")}
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 7.8), sharex=True, sharey="row")
    panels = [(cohort, task) for cohort in ["nki", "bnu1"] for task in ["linear", "nonlinear"]]
    for column, (cohort, task) in enumerate(panels):
        panel = result.loc[result.cohort.eq(cohort) & result.task.eq(task)]
        axes[0, column].set_title(
            ("NKI" if cohort == "nki" else "Beijing") + " · " +
            ("Linear recall" if task == "linear" else "Three-lag product"), fontsize=11, loc="left"
        )
        for rung, (_, color, marker) in styles.items():
            curve = panel.loc[panel.rung.eq(rung)].sort_values("alpha")
            if curve.empty:
                continue
            exponent = np.r_[np.log10(curve.alpha.iloc[:-1]), 10]
            for row, measure in enumerate(["target_r2_median", "target_r2_median", "fraction_people_positive"]):
                axes[row, column].plot(exponent, curve[measure], color=color, marker=marker,
                                       linestyle="--" if rung == "numerical_changed_only" else "-",
                                       linewidth=1.25, markersize=2.7)
        for ax in axes[:, column]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", lw=.5, alpha=.25)
            ax.tick_params(labelsize=8)
    axes[0, 0].set_yscale("symlog", linthresh=1)
    axes[0, 0].set_yticks([-1e12, -1e8, -1e4, -1, 0, 1], ["−10¹²", "−10⁸", "−10⁴", "−1", "0", "1"])
    axes[0, 0].set_ylabel("Median target R²\nFull range")
    axes[1, 0].set_ylim(-.02, .60)
    axes[1, 0].set_ylabel("Median target R²\nDetail")
    axes[2, 0].set_ylim(-.025, 1.025)
    axes[2, 0].set_ylabel("People with target R² > 0")
    axes[2, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[2, 0].set_xticks([-14, -10, -6, -2, 2, 6, 10], ["−14", "−10", "−6", "−2", "2", "6", "∞"])
    fig.suptitle("How consistently does the same decoder transfer across people?", x=.09, y=.985,
                 ha="left", fontsize=16)
    legend = [Line2D([], [], color=color, marker=marker, markersize=4, label=label,
                     linestyle="--" if rung == "numerical_changed_only" else "-")
              for rung, (label, color, marker) in styles.items()]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.52, .944), frameon=False, ncol=5,
               fontsize=9)
    fig.supxlabel("Relative ridge penalty (log₁₀ α), with ∞ marking the exact training-mean predictor",
                  y=.06, fontsize=10)
    fig.text(.09, .023, "Each person has equal weight (NKI 25, Beijing 57). Reruns and direction halves are averaged "
             "within each person before computing the median or the fraction above zero.", fontsize=9)
    fig.subplots_adjust(left=.09, right=.989, top=.86, bottom=.13, wspace=.19, hspace=.24)
    for ext in ["png", "pdf"]:
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(OUT / f"robust_participant_paths.{ext}", dpi=160, metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    participant, matrices = participant_scores()
    paths = summarize(participant)
    paths.to_csv(OUT / "robust_participant_summary.csv", index=False)
    summarize_matrices(matrices).to_csv(OUT / "robust_numerical_matrices.csv", index=False)
    plot(paths)
