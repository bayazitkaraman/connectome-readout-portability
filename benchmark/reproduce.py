"""Replay the main study in a separate working directory, using archived or fetched matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CALIBRATION = ["results/nki-calibration/scores.csv","results/bnu1-calibration/scores.csv"]
SCALE_MANIFESTS = [f"results/pipeline-robustness/schaefer{s}_manifest.json" for s in (200,300)]
BASE_INPUTS = ["nki/nki_reference.npz","nki/inputs.h5","nki/crosswalk.csv",
               "replication_candidates/corr/bnu1_dkt_retest.npz",
               "replication_candidates/corr/bnu1_schaefer200.npz","replication_candidates/corr/bnu1_schaefer300.npz"]


def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)


def digest(path,algorithm="sha256"):
    h=hashlib.new(algorithm)
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def fetch(url,dest,algorithm,expected):
    dest.parent.mkdir(parents=True,exist_ok=True)
    part=dest.with_suffix(".download")
    try:
        with urllib.request.urlopen(url,timeout=90) as response, part.open("wb") as handle:
            shutil.copyfileobj(response,handle)
        if digest(part,algorithm)!=expected:
            raise ValueError(f"Checksum mismatch for {dest.name} from its published source.")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def execute(out,*args):
    print("Running",sys.executable,*map(str,args),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=out,check=True)


def prepare(out,local):
    # An explicit checkout of scripts and input references isolates every output
    # from the archived study results and from a user's source-data directory.
    for name in ("vendor","benchmark","scripts"):
        shutil.copytree(ROOT/name,out/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    copy(ROOT/"run_feasibility.py",out/"run_feasibility.py")
    for rel in CALIBRATION+SCALE_MANIFESTS+["results/numerical-repeat/PROTOCOL.md","results/portability-baseline/protocol.md",
                                            "results/pipeline-robustness/PROTOCOL.md","NONLINEAR_RECALL_PROTOCOL.md"]:
        copy(ROOT/rel,out/rel)
    if local:
        for rel in BASE_INPUTS:
            copy(local/rel,out/"data"/rel)
    else:
        sys.path.insert(0,str(SCRIPTS))
        import prepare_nki as nki
        d=out/"data/nki"
        fetch(nki.SOURCE,d/"inputs.h5","md5",nki.MD5)
        fetch(nki.CROSSWALK,d/"crosswalk.csv","sha256",nki.CROSSWALK_SHA)
        execute(out,"scripts/prepare_nki.py","--source",d/"inputs.h5","--crosswalk",d/"crosswalk.csv")
        execute(out,"scripts/prepare_corr_retest.py")
        execute(out,"scripts/prepare_bnu_scale.py")
    # Cross-check numeric input contents using stored provenance (archive
    # compression/ZIP timestamps can differ across systems).
    import numpy as np
    ni=json.loads((ROOT/"data/nki/provenance.json").read_text(encoding="utf-8"))
    bn=json.loads((ROOT/"data/replication_candidates/corr/bnu1_source_manifest.json").read_text(encoding="utf-8"))
    with np.load(out/"data"/BASE_INPUTS[0],allow_pickle=False) as f:
        assert hashlib.sha256(np.asarray(f["weights"],dtype="<f8").tobytes()).hexdigest()==ni["weights_sha256"]
    assert digest(out/"data"/BASE_INPUTS[1])==ni["source_sha256"]
    assert digest(out/"data"/BASE_INPUTS[2])==ni["crosswalk_sha256"]
    # The published Beijing extraction is pinned by hash and the extraction
    # script also verifies each downloaded CSV against the source listing.
    with np.load(out/"data"/BASE_INPUTS[3],allow_pickle=False) as f:
        if digest(out/"data"/BASE_INPUTS[3])!=bn["data_sha256"]:
            # Reconstruct each canonical UTF-8 source edge list's numeric rows
            # from its verified downloaded copies if compression changes.
            raw=out/"data/replication_candidates/corr/raw"
            if local: raw=local/"replication_candidates/corr/raw"
            for record in bn["files"]:
                path=raw/Path(record["key"]).name
                if digest(path)!=record["sha256"]:
                    raise ValueError(f"Cannot verify Beijing source {path}")
                v=f["sessions"].astype(str).tolist().index(record["session"])
                i=f["subjects"].astype(str).tolist().index(record["subject"])
                edges=np.loadtxt(path,ndmin=2,dtype=int)
                expected=np.zeros(f["weights"].shape[-2:],dtype=int)
                a,b,w=edges.T
                expected[a-1,b-1]=expected[b-1,a-1]=w
                np.testing.assert_array_equal(expected,f["weights"][v,i])
    for s in (200,300):
        manifest=json.loads((ROOT/f"results/pipeline-robustness/schaefer{s}_manifest.json").read_text())
        with np.load(out/f"data/replication_candidates/corr/bnu1_schaefer{s}.npz",allow_pickle=False) as f:
            assert hashlib.sha256(f["weights"].tobytes()).hexdigest()==manifest["weights_sha256"]


def replay(out):
    for script in ("frozen_single_donors","nki_split_half","transfer_grid_audit","transfer_donor_checks",
                   "transfer_similarity_audit"):
        execute(out,f"scripts/{script}.py")
    for grid in ("archived","cross_tuned","dense","fixed10"):
        execute(out,"scripts/source_self_reference.py","--grid",grid)
    for script in ("nonlinear_recall","nonlinear_core_boundary","verify_nonlinear_recall","pipeline_robustness",
                   "pipeline_retuning","nki_numerical_repeat","summarize_benchmark_tasks"):
        execute(out,f"scripts/{script}.py")
    execute(out,"-m","benchmark.study_reliability")
    for script in ("run_visit_portability_path", "run_rung_portability_path",
                   "summarize_portability_path", "verify_portability_path", "check_matched_retention",
                   "robust_portability_path", "source_state_noise", "retention_uncertainty",
                   "state_shift_visits", "state_shift_technical", "summarize_state_shifts",
                   "describe_trajectory_pairs", "unlabeled_parcel_alignment"):
        execute(out, f"scripts/{script}.py")


def verify(out):
    import pandas as pd
    import numpy as np
    checks = {"results/portability-path/path_summary.csv":["source_r2","target_r2","gap_r2"],
              "results/portability-path/shares_summary.csv":["own_share_pct","stranger_share_pct",
                  "own_minus_stranger_pp","own_minus_stranger_pp_lo","own_minus_stranger_pp_hi"],
              "results/portability-path/shift_summary.csv":["shift_pct","target_r2","isotropic_r2",
                  "graph_minus_iso","graph_minus_iso_lo","graph_minus_iso_hi","n_isotropic_positive","n_graph_positive"],
              "results/portability-path/trajectory_summary.csv":["displacement_pct","pattern_r","amplitude_ratio",
                  "mean_shift_pct","people","pairs"],
              "results/portability-path/trajectory_contrasts.csv":["targets","own_minus_stranger","ci_low","ci_high"],
              "results/portability-baseline/summary.csv":["n_people","source_r2","untouched_r2","aligned_r2",
                  "delta_r2","delta_lo","delta_hi","untouched_retention","aligned_retention",
                  "n_untouched_positive","n_aligned_positive","n_improved"],
              "results/portability-path/matched_retention_summary.csv":["source_r2","target_r2","ratio_of_means","ratio_lo","ratio_hi"],
              "results/portability-path/robust_participant_summary.csv":["target_r2_median","n_people_positive","n_omit_two_means_peaking_here"],
              "results/portability-path/robust_numerical_matrices.csv":["median_matrix_r2","n_matrices_positive"],
              "results/portability-path/noise_summary.csv":["mean_r2","median_r2","n_positive"],
              "results/portability-path/noise_envelope.csv":["mean_r2"],
              "results/portability-path/noise_modes.csv":["median_low_slope_energy_share","median_low_fitted_variance_share"],
              "results/portability-path/nearest_vs_own.csv":["nearest_minus_own","lo","hi"],
              "results/nonlinear-recall/summary.csv":["mean"],
              "results/numerical-repeat/summary.csv":["mean"],
              "results/transfer-similarity/grid_summary.csv":["dense_own","coarse_own"],
              "results/pipeline-robustness/task_comparison_summary.csv":["mean"],
              "results/pipeline-robustness/fitted_score_reliability.csv":["spearman","icc_absolute_single"],
              "results/pipeline-robustness/summary.csv":["gain"],
              "results/pipeline-robustness/retuned_summary.csv":["mean"],
              "results/nonlinear-recall/boundary_summary.csv":["mean"]}
    errors={}
    for rel,columns in checks.items():
        new=pd.read_csv(out/rel)
        expected=pd.read_csv(ROOT/rel)
        if list(new.columns)!=list(expected.columns) or len(new)!=len(expected):
            raise ValueError(f"Unexpected dimensions during replay of {rel}")
        if rel.startswith(("results/portability-path/","results/portability-baseline/")):
            keys = [col for col in ("cohort","source_set","task","rung","alpha","matrix_status","noise_fraction","metric") if col in expected]
            pd.testing.assert_frame_equal(new[keys], expected[keys])
        for col in columns:
            np.testing.assert_allclose(new[col],expected[col],atol=2e-8,rtol=2e-8,equal_nan=True)
        errors[rel]={c:float(np.max(np.abs(new[c]-expected[c]))) for c in columns}
    (out/"replay_verification.json").write_text(json.dumps(errors,indent=2)+"\n",encoding="utf-8")
    print(f"Verified archived cohort summaries in {out}.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("workdir",type=Path,help="new, separate directory for replay")
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fetch",action="store_true",help="download published inputs from their original sources")
    source.add_argument("--data-root",type=Path,help="existing original study data tree with verified matrices and NKI HDF")
    a=p.parse_args()
    out=a.workdir.resolve()
    if out.exists():
        p.error("Choose a new replay directory; existing directories are preserved.")
    if any((ROOT/f)==out or (ROOT/f) in out.parents for f in ("scripts","benchmark","vendor","data","results")):
        p.error("Choose a work directory outside the source, data and archived-results directories.")
    out.mkdir(parents=True)
    prepare(out,a.data_root.resolve() if a.data_root else None)
    replay(out)
    verify(out)


if __name__=="__main__":
    main()
