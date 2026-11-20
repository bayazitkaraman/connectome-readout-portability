"""Draw the article's equal-budget calibration curves from cohort summaries."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHODS = [("frozen_r2", "Frozen", "#404b4e", "o", "--"),
           ("aligned_r2", "Unlabeled alignment", "#006a91", "s", "-"),
           ("labeled_r2", "Alignment + labeled correction", "#b64d11", "^", "-")]


def decorate(ax, title, ordinate):
    ax.spines[["right","top"]].set_visible(False)
    ax.spines[["bottom","left"]].set_color("#68777d")
    ax.tick_params(length=3, width=.65, pad=2)
    ax.set_title(title, loc="left", fontsize=9.2)
    ax.set_ylabel(ordinate)
    ax.axhline(0, color="#c0c8ca", lw=.7, zorder=0)


def budget_axis(ax, budgets):
    positions = np.arange(len(budgets))
    ax.set_xticks(positions, [f"{int(n):,}" for n in budgets], rotation=40, ha="right")
    ax.set_xlim(-.23, len(budgets)-.76)
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results"/"calibration-budget-curve")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.results/"summary.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.3, "axes.labelsize":8.5,
                         "axes.linewidth":.7, "xtick.labelsize":7.4, "ytick.labelsize":8,
                         "axes.unicode_minus":True, "pdf.fonttype":42})
    fig, axes = plt.subplots(2, 2, figsize=(6.75,5.65))
    fig.subplots_adjust(left=.100, right=.985, top=.925, bottom=.165, hspace=.66, wspace=.37)
    for j, (cohort, label, n) in enumerate([("nki","NKI",25),("bnu1","Beijing",57)]):
        group = summary.loc[summary.cohort.eq(cohort)].sort_values("budget")
        np.testing.assert_array_equal(group.budget, [0,8,16,32,64,128,256,512,1024,2352])
        assert group.n_people.eq(n).all()
        ax = axes[0,j]
        decorate(ax, f"({'ab'[j]})  {label} ({n} participants)", "Held-out repeat R²")
        xs = budget_axis(ax, group.budget)
        for metric, _, color, mark, lstyle in METHODS:
            ax.plot(xs, group[metric], color=color, marker=mark, ms=3.1, lw=1.25,
                    linestyle=lstyle, markeredgewidth=.35)
            assert group[metric].between(-.58,1.0).all()
        ax.set_ylim(-.58,1.02)
        ax.set_yticks([-.5,0,.5,1.])
        bottom = group.loc[group.budget >= 32]
        ax = axes[1,j]
        decorate(ax, f"({'cd'[j]})  {label} alignment gain", "Aligned minus frozen R²")
        xs = budget_axis(ax, bottom.budget)
        lo = bottom.aligned_minus_frozen_lo.to_numpy()
        hi = bottom.aligned_minus_frozen_hi.to_numpy()
        assert min(lo)>-.14 and max(hi)<.085
        ax.fill_between(xs,lo,hi,color="#006a91",alpha=.20,lw=0)
        ax.plot(xs,bottom.aligned_minus_frozen,color="#006a91",marker="s",lw=1.35,ms=3.6)
        ax.set_ylim(-.14,.085)
        ax.set_yticks([-.10,-.05,0,.05])
        ax.set_xlabel("Target calibration states")
    handles = [Line2D([0],[0],color=c,marker=m,ls=style,ms=4,lw=1.3,label=name)
               for _,name,c,m,style in METHODS]
    fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False, fontsize=8.0,
               handlelength=2, columnspacing=1.2, handletextpad=.45, bbox_to_anchor=(.5,.010))
    fig.savefig(out/"Figure_4.pdf", metadata={"Creator":"Study plotting script","CreationDate":None,
                                              "ModDate":None})
    fig.savefig(out/"Figure_4.png", dpi=600, metadata={"Software":"Study plotting script"})
    plt.close(fig)


if __name__ == "__main__":
    main()
