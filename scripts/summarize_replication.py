"""Summarize a new cohort with the original participant-level calibration estimands."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_nki_calibration import analyze_condition, crossings


KEYS = ["source_visit", "target_visit", "subject", "repeat", "budget", "method"]
NKI = Path(__file__).resolve().parents[1]/"results"/"nki-calibration"


def save(sets, destination):
    for name in sets[0]:
        data = pd.concat([pd.DataFrame(s[name]) for s in sets], ignore_index=True)
        data.to_csv(destination/(name+".csv"),index=False)


def panel(ax, curves, title):
    table = curves.pivot(index="budget",columns="method",values="mean_r2").sort_index()
    budgets = table.index.to_numpy()
    # Give the zero-label baseline its own slot one tick before eight.
    positions = np.where(budgets==0,2,np.log2(np.maximum(budgets,1)))
    for col,label,color,style,marker in [
        ("personal","Personal first scan","#006f71","-","o"),
        ("pooled","Pooled first scans","#b3542b","--","s"),
        ("scratch","Target only","#65517c",":","^")]:
        ax.plot(positions,table[col],label=label,color=color,ls=style,marker=marker,
                lw=1.8,ms=4,mfc="white" if col!="personal" else color)
    ax.axhline(.5,color="#8a939b",ls="--",lw=.8)
    personal, pooled = crossings(table[["personal","pooled"]].to_numpy().T, budgets, .5)
    if np.isfinite(personal) and personal == pooled and personal > 0:
        ax.annotate(f"Both sustain R² ≥ 0.50\nat {personal:.0f} steps",
                    xy=(np.log2(personal), .5), xytext=(20,-34),
                    textcoords="offset points", fontsize=8, va="top", color="#34454c",
                    arrowprops=dict(arrowstyle="-", color="#61727a", lw=.8))
    ticks = [k for k in [0,8,16,32,64,128,256,512,2352] if k in budgets]
    ax.set_xticks([np.log2(t) if t else 2 for t in ticks],
                  [f"{t:,}" for t in ticks],fontsize=8,rotation=35)
    ax.set_title(title,loc="left",fontsize=11,fontweight="bold")
    ax.set_xlabel("Labeled target time steps")
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(axis="y",lw=.5,alpha=.22)
    ax.set_axisbelow(True)
    return table


def make_figure(c, name, n, folder):
    old = pd.read_csv(NKI/"refinement"/"curves.csv")
    fig, axes = plt.subplots(1,2,figsize=(10.8,4.7),sharey=True)
    panel(axes[0],old,"NKI–Rockland (25 people)")
    panel(axes[1],c,f"{name} ({n} people)")
    axes[0].set_ylabel("Mean held-out R²")
    fig.subplots_adjust(left=.080,right=.982,top=.74,bottom=.20,wspace=.13)
    fig.suptitle("Recall after calibrating a decoder on the second scan",x=.080,y=.97,
                 ha="left",fontsize=16,fontweight="bold")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper left",bbox_to_anchor=(.072,.91),frameon=False)
    fig.text(.080,.04,"Each point averages ten delays and six calibration draws per person. Dashed horizontal line marks R² = 0.50.",
             fontsize=8.5,color="#3c4d58")
    fig.savefig(folder/"calibration_comparison.png",dpi=215,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def run(folder, source, name):
    with np.load(source,allow_pickle=False) as data:
        ids=set(data["subjects"].astype(str))
    n=len(ids)
    coarse=pd.read_csv(folder/"scores.csv",dtype={"subject":str})
    assert set(coarse.subject)==ids and set(zip(coarse.source_visit,coarse.target_visit))=={(1,2),(2,1)}
    groups=[]
    for (sv,tv),s in coarse.groupby(["source_visit","target_visit"]):
        assert set(s.subject)==ids
        groups.append(analyze_condition(s,dict(source_visit=sv,target_visit=tv),expected_n=n))
    save(groups,folder)
    primary=coarse[coarse.source_visit==1].copy()
    rf=folder/"refinement"
    fwd=groups[0]
    if (rf/"scores.csv").exists():
        new=pd.read_csv(rf/"scores.csv",dtype={"subject":str})
        assert set(new.subject)==ids and set(new.source_visit)=={1} and not new.duplicated(KEYS).any()
        original_budgets=set(primary.budget)
        coarse_table=primary.groupby(["method","budget"]).r2.mean().unstack()
        base=coarse_table.columns.to_numpy()
        limits=crossings(coarse_table.loc[["personal","pooled"]].to_numpy(),base,.5)
        required=set()
        for upper in limits.astype(int):
            if upper>0:
                required.update(range(int(base[base<upper][-1]),int(upper)+1,4))
        assert required-original_budgets <= set(new.budget)
        overlap=new.merge(primary,on=KEYS,validate="one_to_one",suffixes=("_new","_old"))
        if len(overlap):
            for k in ("r2","nmse"):
                np.testing.assert_allclose(overlap[k+"_new"],overlap[k+"_old"],atol=1e-9,rtol=1e-9)
        primary=pd.concat([primary,new[~new.budget.isin(original_budgets)]],ignore_index=True)
        fwd=analyze_condition(primary,dict(source_visit=1,target_visit=2),expected_n=n)
        save([fwd],rf)
    curves=pd.DataFrame(fwd["curves"])
    # Recompute sustained crossings by explicit iteration over a separate grouping.
    for method,d in primary.groupby("method"):
        curve=d.groupby("budget").r2.mean().sort_index()
        first=next((b for b in curve.index if curve.loc[b:].min()>=.5),np.nan)
        row=next(row for row in fwd["thresholds"] if row["threshold"]==.5 and row["method"]==method)
        np.testing.assert_allclose(first,row["cohort_first_sustained"],equal_nan=True)
    make_figure(curves,name,n,folder)
    print(f"Checked {n} subject identities and the forward sustained crossings on {primary.budget.nunique()} budgets.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results",type=Path,required=True)
    cli.add_argument("--data",type=Path,required=True)
    cli.add_argument("--name",required=True)
    args=cli.parse_args()
    run(args.results,args.data,args.name)
