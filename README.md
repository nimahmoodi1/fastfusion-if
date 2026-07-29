# FastFusion-IF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21652863.svg)](https://doi.org/10.5281/zenodo.21652863)

**Partner-blind protein–protein interface prediction from molecular surface and atomic geometry.**

FastFusion-IF predicts, for every residue of a *single* protein chain, the probability that it
lies on a protein–protein interface — **without being told the binding partner**. It fuses three
complementary views of the chain:

1. an **atomic E(n)-equivariant graph network (EGNN)** over heavy atoms,
2. a **mesh-free solvent-excluded surface point cloud** (no MSMS-style triangulated meshing),
3. **per-residue evolutionary and structural profiles** (PSSM + HMM + DSSP + resAF, 61 dims),

combined by **bidirectional cross-modal attention**, aggregated by **attention residue pooling**,
refined by a **residue-context graph transformer**, and read out through a **gated decoder**.

The model has **≈1.39 M parameters** and trains end-to-end on a single **8 GB** consumer GPU.
Predictions are **invariant to the orientation of the input structure by construction** — every
module consumes only interatomic distances.

---

## Headline results

AGAT-PPIS benchmark, three-seed ensemble. **Protocol B** (leakage-free: the
validation split is held out from Train_335, so no test set informs model
selection) is the primary protocol and matches the manuscript. Protocol A
follows the convention used by published methods, in which Test_60 is the
selection set; under it, Test_60 and its subset Btest_25 are not independent
estimates.

| Test set | Protocol B (AUPRC / MCC) | Protocol A (AUPRC / MCC) |
|---|---|---|
| Test_60 | 0.564 / 0.450 | 0.578 / 0.466 * |
| Test_315-28 | 0.528 / 0.442 | 0.544 / 0.456 |
| Btest_25 | 0.523 / 0.424 | 0.529 / 0.440 * |
| UBtest_25 | **0.367 / 0.320** | 0.404 / 0.339 |

\* selection-set data under Protocol A; not an independent estimate.

Under Protocol B, FastFusion-IF attains the **best AUPRC of all evaluated
methods on the unbound test set** (0.367 versus 0.354 for the previous best,
RGCNPPIS) and is competitive with AGAT-PPIS on the bound benchmarks, at 1.39M
parameters. Reproduce both protocols with `scripts/prepare_benchmark.py
--protocol paper` and `--protocol holdout`; see RESULTS.md for full metrics.

---

## Repository layout

```
fastfusion_if/          the model + data pipeline (Python package)
  ├─ models/            EGNN, surface encoder, cross-modal fusion, pooling, decoder
  └─ data/              PDB parsing, surface sampling, graphs, caching, labels, ESM
scripts/                command-line entry points
configs/                JSON experiment configs (model + training hyperparameters)
manifests/benchmark/    the exact train/val/test split definitions used in the paper
checkpoints/            released model weights (see "Pretrained checkpoints")
RESULTS.md              full result tables
requirements.txt        Python dependencies
pyproject.toml          package metadata
```

Caches, training runs and evaluation dumps are **not** in the repository — they are large and
fully regenerable by the steps below.

---

## Pretrained checkpoints

Two checkpoints are released, and **they are not interchangeable**. Please read this section
before running inference; the difference determines whether you need a heavy feature pipeline.

| File | Inputs | Test_315-28 AUPRC | Needs external features? |
|---|---|---|---|
| `checkpoints/fastfusion_if_full_seed42.pt`<br>(+ `_seed1`, `_seed2`) | atom + surface + profiles | **0.544** (3-seed ens.) | **Yes** — 61-dim PSSM/HMM/DSSP/resAF |
| `checkpoints/fastfusion_if_geometry_only.pt` | atom + surface | 0.395 (3-seed ens.) | **No** — runs on any PDB |

**The full model is the paper model.** It reaches 0.544 AUPRC, but it was trained with 61-dim
evolutionary/structural profiles per residue and *requires them at inference time*. Those
profiles come from PSI-BLAST (PSSM), HHblits against UniClust30 (HMM), DSSP (secondary structure
and solvent accessibility) and an atom-feature block (resAF) — collectively tens of GB of
databases and a multi-hour setup. On the benchmark proteins this is free, because AGAT-PPIS ships
the precomputed features; on an arbitrary new PDB you must generate them yourself.

**The geometry-only model needs nothing.** It takes a bare PDB file and produces predictions
immediately. It is meaningfully less accurate (0.395 vs 0.544 AUPRC), but it is a genuinely usable
partner-blind predictor with a zero-dependency install, and it is the right starting point if you
just want to try the method.

> **Note.** `scripts/infer_pdb.py` builds its input from the PDB alone, which yields a 36-dim
> handcrafted residue-feature vector — not the 61-dim profile vector the full model expects.
> Running `infer_pdb.py` with a **full** checkpoint therefore raises a shape error. This is
> intentional: it fails loudly rather than silently predicting from the wrong features. Use the
> geometry-only checkpoint for bare-PDB inference, or supply real profiles via the cached
> benchmark pipeline.

### Inference on any PDB (geometry-only, no setup)

```bash
python scripts/infer_pdb.py \
  --checkpoint checkpoints/fastfusion_if_geometry_only.pt \
  --pdb my_protein.pdb \
  --chain A \
  --out-prefix prediction
```

Writes two files:

- `prediction_my_protein_A.csv` — one row per residue: `chain, res_seq, insertion, res_name, prob_interface`
- `prediction_my_protein_A_bfactor.pdb` — the same structure with the interface probability
  (×100) written into the B-factor column, so you can colour it directly in PyMOL:

```
# in PyMOL
load prediction_my_protein_A_bfactor.pdb
spectrum b, blue_white_red, minimum=0, maximum=100
show surface
```

### Inference with the full model

The full model runs through the cached pipeline, where the 61-dim profiles are attached. Once you
have completed steps 3–5 of the reproduction below, evaluate any manifest split with:

```bash
python scripts/evaluate_ensemble.py \
  --checkpoints checkpoints/fastfusion_if_full_seed42.pt \
                checkpoints/fastfusion_if_full_seed1.pt \
                checkpoints/fastfusion_if_full_seed2.pt \
  --manifest manifests/benchmark/bench_test315.csv --split test \
  --cache-dir cache/bench_evo --threshold 0.63 \
  --out-dir eval/my_eval
```

---

## Reproducing the benchmark result

This is the main path. It needs only the (small) AGAT-PPIS benchmark data — **no ESM-2 download,
no DIPS-Plus**. Budget roughly 1–2 hours of GPU time for all three seeds on an RTX 3060 Ti.

### 1. Requirements

**Hardware.** An NVIDIA GPU with a CUDA 12.1-capable driver. The paper used an **RTX 3060 Ti
(8 GB)**, which is sufficient — the mesh-free surface encoder is what keeps it inside that budget.
16–32 GB system RAM. About 10 GB free disk for the benchmark cache.

**Software.** Linux, [Miniconda](https://docs.conda.io/en/latest/miniconda.html), Python 3.10,
PyTorch 2.4.1 (CUDA 12.1).

### 2. Installation

```bash
git clone https://github.com/nimahmoodi1/fastfusion-if.git
cd fastfusion-if

conda create -n fastfusion python=3.10 -y
conda activate fastfusion

# PyTorch 2.4.1 + CUDA 12.1 — the exact build used for the paper
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
              pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -r requirements.txt
pip install -e .
```

Check the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.4.1 True NVIDIA GeForce RTX 3060 Ti     (or your GPU)
```

Check the package imports (this catches a broken or partial clone):

```bash
python -c "from fastfusion_if.data.dataset import ProteinInterfaceDataset; from fastfusion_if.models import FastFusionIF; print('OK')"
```

### 3. Get the benchmark data

The AGAT-PPIS repository provides the dataset splits, the PDB structures **and** the precomputed
per-residue features — you do not need to run PSI-BLAST, HHblits or DSSP yourself.

```bash
cd ~                                    # or wherever you keep datasets
git clone https://github.com/AILBC/AGAT-PPIS.git
```

You should now have:

```
AGAT-PPIS/Dataset/{Train_335.pkl, Test_60.pkl, Test_315-28.pkl, Test_315.pkl, UBtest_31-6.pkl}
AGAT-PPIS/Dataset/pdb/*.pdb
AGAT-PPIS/Feature/{pssm,hmm,dssp,resAF}/<protein_id>.npy
```

Only `pssm` (20) + `hmm` (20) + `dssp` (14) + `resAF` (7) = **61 dims** are used.

### 4. Build the split manifests, labels and profile features

From the repository root, with the `fastfusion` environment active:

```bash
python scripts/prepare_benchmark.py \
  --dataset-dir ~/AGAT-PPIS/Dataset \
  --feature-dir ~/AGAT-PPIS/Feature \
  --config configs/full_v2_cached.json \
  --out-dir manifests/benchmark
```

This writes into `manifests/benchmark/`:

| File | What it is |
|---|---|
| `bench_train.csv` | train (335) + val (60) + test (287) in one manifest |
| `bench_test315.csv` | Test_315-28, 287 chains |
| `bench_ubtest.csv` | UBtest_25, 25 chains |
| `bench_test60.csv` | Test_60, the validation set |
| `bench_all.csv` | all 735 chains, split column `all` — used for cache building |
| `bench_labels.json` | the benchmark's own per-residue interface labels |
| `bench_evo.npz` + `bench_evo_keys.json` | the standardized 61-dim profile features |
| `bench_alignment_report.csv` | per-protein label-alignment diagnostics |

Expect it to report `evo features: 735 proteins, dim=61` and `train=335 val=60 test=287`.

> The manifest CSVs in this repository record the exact splits used in the paper. Regenerating
> them reproduces the same membership, since the split is defined by the benchmark, not sampled.

### 5. Build the feature cache

One command. The cache stores parsed atoms, the sampled surface point cloud, all graphs, the
benchmark labels and the 61-dim profiles, so training never recomputes geometry.

```bash
python scripts/precompute_cache.py \
  --manifest manifests/benchmark/bench_all.csv \
  --config configs/bench_evo_reg.json \
  --cache-dir cache/bench_evo \
  --splits all \
  --labels-file manifests/benchmark/bench_labels.json \
  --evo-file manifests/benchmark/bench_evo.npz
```

> **`--labels-file` is mandatory.** The benchmark PDBs are single chains with the partner removed,
> so the geometric 5 Å labelling rule cannot fire — the labels must come from the benchmark's own
> `.pkl` files. Omitting this flag silently produces an all-negative cache and training will not
> learn anything.

Sanity-check that every cached chain carries a uniform 61-dim profile channel:

```bash
python - <<'PY'
import pickle, glob
dims, n = set(), 0
for f in glob.glob("cache/bench_evo/**/*.pkl", recursive=True):
    for ex in pickle.load(open(f, "rb")):
        dims.add(int(ex.residue_features.shape[1])); n += 1
    if n >= 400: break
print("residue_features widths:", sorted(dims), "(must be exactly [61])")
PY
```

### 6. Train the three seeds

The paper model is **atom + surface + profiles** (`configs/bench_evo_reg.json`). ESM-2 is
deliberately **off** here — at 335 training proteins it hurts (see "Ablations").

```bash
# seed 42 (the config default — no --seed flag)
python scripts/train.py --manifest manifests/benchmark/bench_train.csv \
  --config configs/bench_evo_reg.json --out-dir runs/bench_evo_pp \
  --cache-dir cache/bench_evo --num-workers 4 --auto-resume

# seed 1
python scripts/train.py --manifest manifests/benchmark/bench_train.csv \
  --config configs/bench_evo_reg.json --out-dir runs/bench_evo_pp_s1 \
  --cache-dir cache/bench_evo --num-workers 4 --seed 1 --auto-resume

# seed 2
python scripts/train.py --manifest manifests/benchmark/bench_train.csv \
  --config configs/bench_evo_reg.json --out-dir runs/bench_evo_pp_s2 \
  --cache-dir cache/bench_evo --num-workers 4 --seed 2 --auto-resume
```

Each run trains 40 epochs, keeps the checkpoint with the best validation PR-AUC as `best.pt`, and
auto-evaluates Test_315-28 at the end. `--auto-resume` makes an interrupted run continue from its
last epoch instead of restarting.

Per-seed AUPRC on Test_315-28 should land near **0.533 / 0.530 / 0.539**.

### 7. Evaluate the ensemble

First tune the decision threshold on the validation split, then apply it to both test sets.
AUROC and AUPRC are threshold-free; F1 and MCC are not.

```bash
CKPT="runs/bench_evo_pp/best.pt runs/bench_evo_pp_s1/best.pt runs/bench_evo_pp_s2/best.pt"

# tune on validation (Test_60, carried inside bench_train.csv)
python scripts/evaluate_ensemble.py --checkpoints $CKPT \
  --manifest manifests/benchmark/bench_train.csv --split val \
  --cache-dir cache/bench_evo --out-dir eval/bench_evo_ens_val

THR=$(python -c "import json;print(json.load(open('eval/bench_evo_ens_val/val_ensemble_metrics.json'))['threshold'])")
echo "tuned threshold = $THR"      # 0.63 in the paper

# bound test set
python scripts/evaluate_ensemble.py --checkpoints $CKPT \
  --manifest manifests/benchmark/bench_test315.csv --split test --threshold "$THR" \
  --cache-dir cache/bench_evo --out-dir eval/bench_evo_ens_test315_tuned

# unbound test set
python scripts/evaluate_ensemble.py --checkpoints $CKPT \
  --manifest manifests/benchmark/bench_ubtest.csv --split test --threshold "$THR" \
  --cache-dir cache/bench_evo --out-dir eval/bench_evo_ens_ubtest_tuned
```

Each run writes three files into its output directory:

- `test_ensemble_metrics.json` — global and per-protein-summary metrics
- `test_ensemble_per_residue_predictions.csv` — one row per residue with label and probability
- `test_ensemble_per_protein_metrics.csv` — per-chain ROC-AUC, PR-AUC, precision, recall, F1, MCC

### 8. Expected results

| Test set | AUPRC | AUROC | MCC | F1 | threshold |
|---|---|---|---|---|---|
| Test_315-28 | 0.544 | 0.869 | 0.456 | 0.536 | 0.63 |
| UBtest_25 | 0.404 | 0.798 | 0.339 | 0.418 | 0.63 |

Small differences in the last digit across GPUs, driver versions and PyTorch builds are normal.

> **On test-time augmentation.** `evaluate_ensemble.py` accepts `--tta N`, which averages
> predictions over N random rotations. Because the network is rotation-invariant by construction,
> this changes the global metrics by less than 1e-6 while costing N× the inference time. It is
> left in the tool for models that are not invariant; you do not need it here.

---

## Ablations

### Surface on/off

`configs/bench_evo_nosurf_reg.json` is identical to `bench_evo_reg.json` except
`model.use_surface = false`. It reuses the **same cache**, so no rebuild is needed — train three
seeds into `runs/bench_evo_nosurf_pp{,_s1,_s2}` and evaluate exactly as in step 7, tuning its own
threshold on validation (it comes out at 0.57).

| | AUPRC | AUROC | MCC |
|---|---|---|---|
| surface OFF (atom + profiles) | 0.531 | 0.858 | 0.419 |
| **surface ON (full)** | **0.544** | **0.869** | **0.456** |
| surface OFF → ON, unbound AUPRC | 0.348 → **0.404** | | |

A per-protein paired Wilcoxon signed-rank test finds the **AUROC** improvement on the bound set
statistically significant (p = 0.017, 169/287 proteins improve). The AUPRC gain on the bound set
is **not** significant (p = 0.251); the unbound set is too small (n = 25) to power a test. The
surface helps most exactly where conformation-invariant sequence features are weakest.

### Feature ablation

| Model | AUPRC | AUROC | MCC |
|---|---|---|---|
| atom + surface | 0.395 | 0.801 | 0.324 |
| atom + profiles (no surface) | 0.531 | 0.858 | 0.419 |
| **atom + surface + profiles (full)** | **0.544** | **0.869** | **0.456** |

Evolutionary profiles are the largest single contributor; the surface adds a further gain on top.
Adding ESM-2 embeddings at this data scale *reduces* AUPRC — see below.

---

## Advanced: ESM-2 and large-scale DIPS-Plus training

Not needed to reproduce the headline result. Skip unless you want the language-model variants.

### Adding ESM-2 to the benchmark cache

ESM-2 embeddings are only baked into a cache when `--plm-model` is passed to
`precompute_cache.py` — setting `use_plm_features` in the config is **not** sufficient. Build the
ESM cache first, then attach the profiles by reusing its geometry:

```bash
# geometry + ESM-2 650M (downloads ~2.5 GB of ESM weights on first run)
python scripts/precompute_cache.py \
  --manifest manifests/benchmark/bench_all.csv \
  --config configs/bench_esm_reg.json --cache-dir cache/bench_esm --splits all \
  --labels-file manifests/benchmark/bench_labels.json \
  --plm-model esm2_t33_650M_UR50D --plm-device cuda

# attach profiles, reusing the geometry and ESM above
python scripts/precompute_cache.py \
  --manifest manifests/benchmark/bench_all.csv \
  --config configs/bench_evo_esm_reg.json --cache-dir cache/bench_evo_esm --splits all \
  --from-cache cache/bench_esm \
  --labels-file manifests/benchmark/bench_labels.json \
  --evo-file manifests/benchmark/bench_evo.npz
```

### DIPS-Plus at scale

```bash
# download (~100 GB extracted)
python scripts/download_datasets.py --dataset dips-plus --out-dir data \
  --zenodo-pattern final_raw_dips --extract
#   archive: https://zenodo.org/record/8140981/files/final_raw_dips.tar.gz

# leakage-safe split: MMseqs2 clustering at 30% identity, 80% coverage
conda install -c bioconda -c conda-forge mmseqs2 -y
python scripts/prepare_manifest.py \
  --data-dir data/DIPS-Plus/final/raw \
  --out manifests/dips_plus_mmseqs30.csv \
  --cluster-method mmseqs --identity 0.30 --coverage 0.80 --threads 16

# cache + train + evaluate
python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30.csv \
  --config configs/full_v2_esm_reg.json --cache-dir cache/dips_plus_v2_esm --splits train,val,test \
  --plm-model esm2_t33_650M_UR50D --plm-device cuda

python scripts/train.py --manifest manifests/dips_plus_mmseqs30.csv \
  --config configs/full_v2_esm_reg.json --out-dir runs/full_v2_esm_reg \
  --cache-dir cache/dips_plus_v2_esm --num-workers 8 --auto-resume

python scripts/evaluate.py --checkpoint runs/full_v2_esm_reg/best.pt \
  --manifest manifests/dips_plus_mmseqs30.csv --split test \
  --cache-dir cache/dips_plus_v2_esm --out-dir eval/full_v2_esm_reg
```

At this scale (33,690 training chains, 8,422 test chains) ESM-2 **helps**: PR-AUC rises from
0.507 to 0.571 in an otherwise byte-identical configuration. The benefit of language-model
features is scale-dependent — negative on the 335-protein benchmark, clearly positive here.

---

## Configuration reference

| Config | Modalities | Cache |
|---|---|---|
| `bench_evo_reg.json` | atom + surface + profiles — **the paper model** | `cache/bench_evo` |
| `bench_evo_nosurf_reg.json` | atom + profiles (surface ablation) | `cache/bench_evo` |
| `bench_esm_reg.json` | atom + surface + ESM-2 | `cache/bench_esm` |
| `bench_evo_esm_reg.json` | atom + surface + profiles + ESM-2 | `cache/bench_evo_esm` |
| `full_v2_cached.json` | passed to `prepare_benchmark.py` for parsing settings | — |
| `full_v2_esm_reg.json` | DIPS-Plus large-scale recipe | `cache/dips_plus_v2_esm` |

Key flags inside a config: `model.use_surface`, `model.use_residue_features` (profiles),
`model.use_plm_features` (ESM-2).

Paper hyperparameters (`bench_evo_reg.json`): hidden width 96 throughout; 4 EGNN layers,
3 surface layers, 2 fusion layers, 2 residue-context layers, 4 attention heads; dropout 0.2;
40 epochs; AdamW, lr 1.5e-4, weight decay 0.05, 3 warm-up epochs, gradient clipping 1.0;
batch size 1 chain; mixed precision. Loss is class-weighted BCE (the positive weight is estimated
from the training split, ≈5.2) plus a soft-Dice term at weight 0.2.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastfusion_if.data'`** — the clone is incomplete. Run the
import check in step 2; `ls fastfusion_if/data/` should list 17 `.py` files.

**`RuntimeError: ... normalized_shape=[61] ... got input of size [N, 36]`** — you are running a
**full** checkpoint on features it was not trained with, almost always via `infer_pdb.py` on a
bare PDB. Use `checkpoints/fastfusion_if_geometry_only.pt` instead, or go through the cached
pipeline. See "Pretrained checkpoints".

**Training loss falls but every metric stays near zero** — the cache was built without
`--labels-file`, so all residues are negative. Delete `cache/bench_evo` and rebuild with step 5.

**CUDA out of memory** — batch size is already 1 chain. Reduce `data.max_surface_points` (2048 in
the paper config) or `model.fusion_dim` in the config. The benchmark fits in 8 GB as shipped.

**`--from-cache dir does not exist`** — that flag reuses an existing cache; drop it if you are
building from scratch. It is only needed for the ESM path.

---

## Citation

If you use this code or the released checkpoints, please cite the FastFusion-IF paper (details to
follow on publication), and the benchmark and dataset sources as appropriate:

- **AGAT-PPIS** — Zhou, Jiang & Yang, *Briefings in Bioinformatics* 24(3), bbad122, 2023. Source of the benchmark splits, structures and features.
- **GraphPPIS** — Yuan et al., *Bioinformatics* 38(1):125–132, 2022.
- **RGCNPPIS** — Zhong et al., *IEEE/ACM TCBB* 21(6):1676–1684, 2024.
- **GTE-PPIS** — Wang et al., *Briefings in Bioinformatics* 26(3), bbaf290, 2025. Source of the unified baseline numbers quoted above.
- **DIPS-Plus** — Morehead et al., *Scientific Data* 10:509, 2023.
- **ESM-2** — Lin et al., *Science* 379:1123–1130, 2023.

### Software citation

Mahmoudi, N. (2026). FastFusion-IF (Version v1.0.0) [Computer software].
Zenodo. https://doi.org/10.5281/zenodo.21652863


## License

MIT — see [`LICENSE`](LICENSE).
