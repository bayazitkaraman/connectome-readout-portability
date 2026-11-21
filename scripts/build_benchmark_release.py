"""Create a checksummed, GitHub-ready source directory and ZIP from an explicit resource list."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RESULT = {".csv",".md",".json",".png",".txt",".gz",".pdf"}
REQUIRED = (
    "results/portability-path/path_summary.csv",
    "results/portability-path/accuracy_portability_detail.png",
    "results/portability-path/accuracy_portability_full.png",
    "results/portability-path/visit_pairs.csv.gz","results/portability-path/rung_pairs.csv.gz",
    "results/portability-path/README.md","results/portability-path/checks.json",
    "results/portability-path/matched_retention_summary.csv","results/portability-path/noise_summary.csv",
    "results/portability-path/noise_source_paths.pdf","results/portability-path/robust_participant_paths.pdf",
    "results/portability-path/robust_participant_summary.csv",
    "results/portability-path/shares_summary.csv","results/portability-path/shift_summary.csv",
    "results/portability-path/shift_technical_pairs.csv.gz","results/portability-path/shift_visit_pairs.csv.gz",
    "results/portability-path/shift_matched_noise.pdf",
    "scripts/plot_manuscript_figure.py","results/portability-path/Figure_1.pdf","results/portability-path/Figure_1.png",
    "scripts/describe_trajectory_pairs.py","results/portability-path/trajectory_summary.csv",
    "results/portability-path/trajectory_contrasts.csv","results/portability-path/trajectory_people.csv",
    "results/portability-path/trajectory_pairs.csv.gz","results/portability-path/trajectory_checks.json",
    "results/nonlinear-recall/independent_verification.json",
    "results/numerical-repeat/changed_only.json",
    "results/pipeline-robustness/task_comparison.png",
    "results/pipeline-robustness/linear_pooled_replay.csv",
    "results/pipeline-robustness/task_similarity_curves.csv",
    "results/pipeline-robustness/retuned_summary.csv",
    "results/pipeline-robustness/fitted_score_reliability.csv",
    "results/pipeline-robustness/README.md",
    "CITATION.cff","LICENSE","LICENSE-data.txt",".zenodo.json",
    "scripts/plot_manuscript_extension.py","scripts/source_augmentation.py","scripts/summarize_alignment_retention.py",
    "scripts/headline_robustness.py","scripts/calibration_budget_curve.py","scripts/reviewer_context.py",
    "scripts/verify_cli_state_comparisons.py","scripts/plot_calibration_budget.py",
    "results/source-augmentation/summary.csv","results/portability-baseline/retention_intervals.csv",
    "results/headline-robustness/checks.json","results/headline-robustness/summary.csv",
    "scripts/summarize_map_portability.py", "results/headline-robustness/map_portability.csv",
    "results/headline-robustness/reporting_checks.json",
    "results/calibration-budget-curve/checks.json","results/calibration-budget-curve/summary.csv",
    "results/reviewer-context/age_aggregate.csv","results/reviewer-context/spatial_decomposition.csv",
    "results/anatomy-advantage/statistics.csv",
    *(f"results/portability-path/Figure_{i}.{extension}" for i in (2,3) for extension in ("pdf","png")),
    *(f"results/calibration-budget-curve/Figure_4.{extension}" for extension in ("pdf","png")),
    "scripts/regional_noise.py", "scripts/nki_visit_sensitivity.py", "scripts/schaefer200_repeat.py",
    *(f"results/supplementary-tables/S{i}.csv" for i in range(1,10)),
    *(f"results/cli-state-{cohort}/{name}" for cohort in ("nki","beijing")
      for name in ("state_pairs.csv","state_summary.csv","run.json")),
    *(f"results/{cohort}-augmentation/{name}.csv" for cohort in ("nki","bnu1")
      for name in ("contrasts","curves","thresholds")),
    "docs/DATA.md","docs/REPRODUCIBILITY.md","results/FINAL_RESULTS_SUMMARY.md",
    "results/README.md","scripts/README.md","protocols/README.md","protocols/NONLINEAR_RECALL_PROTOCOL.md",
    "scripts/unlabeled_parcel_alignment.py","results/portability-baseline/protocol.md",
    "results/portability-baseline/README.md","results/portability-baseline/people.csv",
    "results/portability-baseline/summary.csv","results/portability-baseline/checks.json",
    "scripts/summarize_benchmark_tasks.py","scripts/pipeline_retuning.py","scripts/prepare_bnu_scale.py")


def members():
    result = {}
    def add(rel):
        p=ROOT/rel
        if not p.is_file():
            raise FileNotFoundError(p)
        result[p.relative_to(ROOT).as_posix()]=p
    for rel in ("run_feasibility.py","requirements-benchmark.txt",
                "benchmark/validation.json",
                "data/nki/provenance.json","data/replication_candidates/corr/bnu1_source_manifest.json",
                "results/nki-calibration/scores.csv","results/bnu1-calibration/scores.csv",
                "results/anatomy-advantage/participants.csv",*REQUIRED):
        add(rel)
    for p in (ROOT/"protocols").glob("*PROTOCOL.md"):
        add(p.relative_to(ROOT))
    for d,extensions in [("benchmark",{".py",".md",".txt"}),("vendor",{".py",".md"}),("scripts",{".py"})]:
        for p in (ROOT/d).rglob("*"):
            if p.is_file() and (p.suffix in extensions or p.name=="LICENSE") and p.name!="RELEASE_README.md":
                add(p.relative_to(ROOT))
    for folder in ("nonlinear-recall","numerical-repeat","pipeline-robustness","transfer-similarity","portability-path",
                   "nki-split-half","density-controls","source-augmentation","portability-baseline",
                   "headline-robustness","reviewer-context","calibration-budget-curve","supplementary-tables",
                   "regional-noise","nki-visit-sensitivity","schaefer200-repeat"):
        for p in (ROOT/"results"/folder).rglob("*"):
            if p.is_file() and p.suffix in ALLOWED_RESULT:
                add(p.relative_to(ROOT))
    for dest,template in (("README.md","benchmark/RELEASE_README.md"),(".gitignore","benchmark/release.gitignore"),
                          (".gitattributes","benchmark/release.gitattributes")):
        p=ROOT/template
        result[dest]=p if p.exists() else ROOT/dest
    return dict(sorted(result.items()))


def build(destination):
    dest=destination.resolve()
    archive=dest.with_suffix(".zip")
    if dest.exists() or archive.exists():
        raise FileExistsError(f"Choose a new destination; archive or directory already exists at {dest}.")
    files=members()
    dest.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=dest.parent,prefix=".release-") as temp:
        stage=Path(temp)/dest.name
        stage.mkdir()
        hashes={}
        for name,src in files.items():
            out=stage/name
            out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,out)
            hashes[name]=hashlib.sha256(out.read_bytes()).hexdigest()
        (stage/"manifest-sha256.json").write_bytes((json.dumps(hashes,indent=2)+"\n").encode("utf-8"))
        tmpzip=Path(temp)/archive.name
        with ZipFile(tmpzip,"x",compression=ZIP_DEFLATED) as compressed:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    entry=ZipInfo(p.relative_to(stage).as_posix(),(1980,1,1,0,0,0))
                    entry.external_attr=0o100644 << 16
                    entry.compress_type=ZIP_DEFLATED
                    compressed.writestr(entry,p.read_bytes())
        with ZipFile(tmpzip) as compressed:
            expected=json.loads(compressed.read("manifest-sha256.json"))
            assert len(compressed.namelist())==len(expected)+1 and compressed.testzip() is None
            for name,sha in expected.items():
                assert hashlib.sha256(compressed.read(name)).hexdigest()==sha
        stage.rename(dest)
        tmpzip.rename(archive)
    print(f"{len(files)} checksummed files, {archive.stat().st_size:,} bytes. {archive}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination",type=Path,default=ROOT/"output/project1-benchmark-release")
    build(p.parse_args().destination)
