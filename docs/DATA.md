# Data for the repeat-connectome benchmark

The study uses measured structural connectivity matrices and generates the recall signals in software. Reproducing the paper requires the source connectomes. The synthetic example generates its own matrices, and the study tables can be inspected directly. [Synthetic input and NPZ schema](../benchmark/README.md) · [Results index](../results/FINAL_RESULTS_SUMMARY.md)

| Source | Study inputs | Where to obtain them |
|---|---|---|
| NKI–Rockland | 25 participants with two visits, two diffusion-direction halves per visit, and 20 perturbed numerical reconstructions per half. The forward visit analysis uses the odd directions from each visit. | [Kiar archive used in this study](https://doi.org/10.5281/zenodo.4041549) · [Visit crosswalk](http://fcon_1000.projects.nitrc.org/indi/enhanced/public_crosswalk_ids_08242023.csv) · [File and matrix checksums](../data/nki/provenance.json) |
| Beijing Normal University, CoRR BNU1 | 57 participants with two visits. The primary analysis uses DKT whole-scan matrices. Schaefer 200 and 300 are used in the separate parcellation analysis. | [Beijing cohort paper](https://doi.org/10.1038/sdata.2015.56) · [m2g data](https://docs.neurodata.io/m2g/paper/data.html) · [DKT source manifest](../data/replication_candidates/corr/bnu1_source_manifest.json) · [Schaefer 200](../results/pipeline-robustness/schaefer200_manifest.json) and [300](../results/pipeline-robustness/schaefer300_manifest.json) manifests |

## Obtain the study inputs

The NKI numerical-rerun count refers to executions. On the first-visit odd half, 20 runs yielded 2–10 distinct off-diagonal graphs per person (median four), and 246 of 500 recovered the reference graph. [Reconstruction counts](../results/reviewer-context/numerical_context.csv)

The legacy NITRC visit-crosswalk host has an invalid certificate for its HTTPS alias. The preparation script retrieves the public file at the provider's HTTP address and verifies the complete SHA-256 against the published-source copy before reading it. The diffusion HDF and Beijing matrices are retrieved over HTTPS. [Pinned hashes](../data/nki/provenance.json)

The study replay with `--fetch` retrieves and verifies the inputs and keeps its prepared copies under the replay directory. For NKI the pinned HDF has 233,049,104 bytes and MD5 `f457d4a860d79654672143c1ace95ee1`. Its numerical reruns are required for the same-acquisition reconstruction comparison. The crosswalk establishes visit order. For Beijing the preparation scripts use the identified source edge lists and atlas manifests. [Reproduction commands](REPRODUCIBILITY.md) · [Provider and software provenance](../benchmark/SOURCES.md)

For offline replay, supply the directory containing this structure to `--data-root`:

```text
nki/
  inputs.h5
  crosswalk.csv
  nki_reference.npz
replication_candidates/corr/
  bnu1_dkt_retest.npz
  bnu1_schaefer200.npz
  bnu1_schaefer300.npz
```

If the DKT NPZ has been repacked and its file checksum differs, the verifier also uses the original checksum-matched edge files in `replication_candidates/corr/raw/`. Participant IDs and source visit labels in the study outputs allow results to be paired with the released inputs. Use each provider’s attribution and access terms for its data.

## Use another cohort

Provide a NumPy archive containing the connectivity array, participant IDs and visit labels. Each parcel index must refer to the same region across all participants and visits. The generic CLI compares ordered visit pairs, including the reverse direction. The paper’s primary analysis uses first visit to second visit. [Array shapes, optional reconstruction axis and output definitions](../benchmark/README.md)
