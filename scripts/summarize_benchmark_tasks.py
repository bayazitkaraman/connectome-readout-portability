"""Generate the journal comparison of single-source, pooled and target-fitted readouts."""
from types import SimpleNamespace

from paired_pool_augmentation import read_graphs, transfer
from pipeline_robustness import BOUNDS, BIN_NAMES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=transfer.ROOT
OUT=ROOT/"results/pipeline-robustness"
pilot=transfer.pilot
ARMS={"local_r2":"Fitted on target","own_r2":"Own first scan","nearest_r2":"Closest other person",
      "mean_other_r2":"Average other person","pooled_r2":"Pooled decoder"}


def collect():
    directory=ROOT/"results"
    product=pd.read_csv(directory/"nonlinear-recall/participant_scores.csv",dtype={"target_id":str})
    linear=pd.read_csv(directory/"transfer-similarity/nearest_donor_people.csv",dtype={"target_id":str})
    panels=[product[["cohort","target_id",*ARMS]].assign(task="product")]
    pool_qc=[]
    for cohort in ("nki","bnu1"):
        graph,ids,half=read_graphs(cohort)
        native=pd.read_csv(directory/f"{cohort}-calibration/scores.csv",dtype={"subject":str})
        pool=native.query("half==@half and source_visit==1 and target_visit==2 and budget==0 and method=='pooled'")
        assert pool.groupby("subject").r2.nunique().eq(1).all()
        pool=pool.groupby("subject").r2.mean()
        group=linear[linear.cohort==cohort].set_index("target_id").loc[ids].copy()
        group["nearest_r2"]=group.selected_r2
        mapping=pilot.input_map(graph.shape[-1],10)
        sources=[]
        for i,g in enumerate(graph[0]):
            net=pilot.reservoir(g,.9)
            ident=10000+i
            sources.append(SimpleNamespace(training=pilot.simulate(net,mapping,ident,0,2352),
                                           validation=pilot.simulate(net,mapping,ident,1,1176)))
        fitted,pooled=[],[]
        for i,g in enumerate(graph[1]):
            net=pilot.reservoir(g,.9)
            ident=10000+len(ids)+i
            model=pilot.RidgeFamily.from_design(pilot.simulate(net,mapping,ident,0,2352))
            alpha=model.tune(pilot.simulate(net,mapping,ident,1,1176),transfer.PENALTIES)
            test=pilot.combine([pilot.simulate(net,mapping,ident,61,512,k) for k in range(4)])
            fitted.append(transfer.errors(model.predict(test[0],alpha),test)[1].mean())
            pooled_fit,pooled_penalty,allocation=transfer.shared_pool(sources,i,ids)
            assert ids[i] not in allocation["final_fit"] and sum(allocation["final_fit"].values())==2352
            pooled.append(transfer.errors(pooled_fit.predict(test[0],pooled_penalty),test)[1].mean())
            pool_qc.append(dict(cohort=cohort,target_id=str(ids[i]),relative_ridge=pooled_penalty,
                                training_steps=2352,pooled_r2=pooled[-1],archived_r2=pool.loc[ids[i]]))
        np.testing.assert_allclose(pooled,pool.loc[ids],rtol=0,atol=1e-9)
        group["pooled_r2"]=pooled
        group["local_r2"]=fitted
        panels.append(group.reset_index()[["cohort","target_id",*ARMS]].assign(task="linear"))
    people=pd.concat(panels,ignore_index=True)
    pd.DataFrame(pool_qc).to_csv(OUT/"linear_pooled_replay.csv",index=False)
    assert len(people)==164 and not people.duplicated(["cohort","task","target_id"]).any()
    people["nearest_minus_mean_other"]=people.nearest_r2-people.mean_other_r2
    people["nearest_minus_pool"]=people.nearest_r2-people.pooled_r2
    people.to_csv(OUT/"task_comparison_people.csv",index=False)
    rows=[]
    for (task,cohort),p in people.groupby(["task","cohort"]):
        draw=np.random.default_rng(20261120).integers(len(p),size=(10000,len(p)))
        for key in [*ARMS,"nearest_minus_mean_other","nearest_minus_pool"]:
            a=p[key].to_numpy()
            lo,hi=np.quantile(a[draw].mean(axis=1),[.025,.975])
            rows.append(dict(task=task,cohort=cohort,measure=key,n=len(a),mean=a.mean(),
                             lo=lo,hi=hi,participants_positive=int((a>0).sum())))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/"task_comparison_summary.csv",index=False)
    return summary


def figure(summary):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),sharey=True)
    fig.subplots_adjust(left=.21,right=.98,bottom=.21,top=.76,wspace=.15)
    for ax,task,title in zip(axes,["linear","product"],["Linear delayed recall","Three-lag product recall"]):
        for c,cohort,color,delta in [("NKI","nki","#076962",.12),("Beijing","bnu1","#a74821",-.12)]:
            p=summary[(summary.task==task)&(summary.cohort==cohort)].set_index("measure").loc[list(ARMS)]
            mean=p["mean"].to_numpy()
            ax.errorbar(mean,np.arange(len(ARMS))[::-1]+delta,xerr=[mean-p.lo,p.hi-mean],
                        fmt="o",color=color,label=c,capsize=2.5,linewidth=1.5,markersize=4.5)
        ax.set_title(title,loc="left",fontweight="bold")
        ax.set_xlim(-.06,1.04)
        ax.axvline(0,color="0.55",linewidth=.8)
        ax.set_axisbelow(True)
        ax.grid(axis="x",color=".9")
        ax.set_xlabel("Mean held-out R²")
        ax.tick_params(length=0)
        ax.spines[["left","bottom"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(ARMS))[::-1],labels=list(ARMS.values()))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.935),ncol=2,frameon=False)
    fig.suptitle("Recall after transferring a decoder to a repeat scan",fontweight="bold")
    fig.supxlabel("95% participant-bootstrap intervals. Target fits use 2,352 fitting and 1,176 tuning labels. Transfer uses zero target labels.",fontsize=8.5)
    fig.savefig(OUT/"task_comparison.png",dpi=180)
    plt.close(fig)


def similarity_by_task():
    fixed=pd.read_csv(OUT/"pair_scores.csv",dtype={"target":str,"source":str}).query("scenario=='native' and not own")
    product=pd.read_csv(ROOT/"results/nonlinear-recall/pair_scores.csv",dtype={"target_id":str,"other_id":str})
    product=product.rename(columns={"other_similarity":"similarity","target_id":"target","other_id":"source","other_r2":"r2"})
    keys=["cohort","target","source"]
    check=fixed.merge(product,on=keys,validate="one_to_one")
    assert len(check)==3792
    np.testing.assert_allclose(check.similarity_x,check.similarity_y,rtol=0,atol=1e-12)
    combined=pd.concat([fixed[keys+["r2","similarity"]].assign(task="linear"),
                        product[keys+["r2","similarity"]].assign(task="product")])
    combined["bin"]=pd.cut(combined.similarity,BOUNDS,labels=BIN_NAMES,right=False)
    stats=combined.groupby(["cohort","task","bin"],observed=True).agg(
        pairs=("r2","size"),target_people=("target","nunique"),mean_r2=("r2","mean"),median_r2=("r2","median"),
        fraction_reaching_010=("r2",lambda x:(x>=.10).mean()),fraction_reaching_050=("r2",lambda x:(x>=.50).mean()))
    stats.to_csv(OUT/"task_similarity_curves.csv")


if __name__=="__main__":
    figure(collect())
    similarity_by_task()
