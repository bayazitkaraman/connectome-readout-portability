"""Plot one error per person and benchmark on the repeat visit."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1]/"results"/"nki-retest"


def main():
    sources = [(OUT,"Penalty selected on the source scan")]
    if (OUT/"transfer-tuned"/"participant_summary.csv").exists():
        sources.append((OUT/"transfer-tuned","Penalty selected on others’ repeat scans"))
    fig, axlist = plt.subplots(1,len(sources),figsize=(5.4*len(sources),4.8),
                               sharey=False,squeeze=False)
    for (folder,title),ax in zip(sources,axlist.ravel()):
        df=pd.read_csv(folder/"participant_summary.csv")
        frame=df[(df.half=="odd")&(df.source_visit==1)].sort_values("subject")
        assert len(frame)==25
        noise=np.random.default_rng(933).uniform(-.19,.19,len(frame))
        measures=[("other_mean_nmse","Other\ndecoders"),("self_nmse","Person’s own\nfrozen decoder"),
                  ("pooled_nmse","Pooled\ndecoder"),("within_scan_nmse","Recalibrated\non target")]
        for j,(col,label) in enumerate(measures):
            y=frame[col].to_numpy()
            assert (y>0).all()
            ax.scatter(j+noise,y,s=18,color="#007a7a" if j==1 else "#46667d",alpha=.59,linewidths=0,zorder=2)
            ax.plot([j-.24,j+.24],[np.median(y)]*2,color="#182a37",lw=2.2,zorder=3)
        ax.axhline(1,color="#ad5229",ls="--",lw=1.4,zorder=1)
        ax.set_xticks(range(4),[label for _,label in measures],fontsize=8.5)
        ax.set_yscale("log")
        ax.set_xlim(-.48,3.48)
        ax.set_title(title,fontsize=11,loc="left")
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
    axlist.ravel()[0].set_ylabel("Mean normalized squared error (lower is better)")
    fig.suptitle("Frozen recall decoders on a second diffusion scan",fontsize=15,x=.07,ha="left",fontweight="bold")
    lines=[Line2D([0],[0],color="#182a37",lw=2),Line2D([0],[0],color="#ad5229",ls="--")]
    fig.legend(lines,["Median across people","Input-variance reference (error = 1)"],
               loc="lower left",bbox_to_anchor=(.064,.06),ncol=2,fontsize=9,frameon=False)
    fig.text(.07,.033,"One point per target. Other decoders show the mean over 24 donors. Panels use different logarithmic scales.",
             fontsize=8.2,ha="left")
    fig.subplots_adjust(left=.08 if len(sources)>1 else .16,right=.985,top=.86,bottom=.24,wspace=.12)
    fig.savefig(OUT/"rescan_error.png",dpi=210,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
