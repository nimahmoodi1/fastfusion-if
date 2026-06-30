# FastFusion-IF

**FastFusion-IF** is a residue-level protein–protein interaction (PPI) interface / binding-site
predictor that fuses four complementary views of a protein chain:

1. an **atomic E(n)-equivariant graph network (EGNN)** over heavy atoms,
2. a **mesh-free solvent-excluded surface point-cloud encoder** (no MSMS-style meshing),
3. **cross-modal attention** that lets atom tokens and surface tokens exchange information,
4. **residue-level attention pooling** feeding a **residue-context graph transformer** and a gated decoder,

with optional **ESM-2** language-model embeddings and **evolutionary profiles** (PSSM/HMM/DSSP/resAF)
injected per residue. The model is ~2.86M parameters and takes single-chain, partner-blind input.

The contribution of the surface modality is that it adds *conformation-aware* geometric signal on top of
conformation-invariant sequence features — most valuable on unbound structures.

---

## Headline results

On the standard AGAT-PPIS benchmark (held-out test sets), 3-seed ensemble with test-time augmentation.
Published baseline values are from GTE-PPIS (Briefings in Bioinformatics 2025, `bbaf290`), Table 3.

**Test_315-28 (287 proteins)**

| Method | MCC | AUPRC |
|---|---|---|
| GraphPPIS | 0.335 | 0.408 |
| RGCNPPIS | 0.352 | 0.420 |
| AGAT-PPIS | 0.442 | 0.525 |
| **FastFusion-IF (ours)** | **0.456** | **0.544** |
| GTE-PPIS (SOTA) | 0.511 | 0.598 |

**UBtest_25 (25 unbound proteins)**

| Method | MCC | AUPRC |
|---|---|---|
| GraphPPIS | 0.298 | 0.330 |
| RGCNPPIS | 0.296 | 0.354 |
| AGAT-PPIS | 0.301 | 0.325 |
| GTE-PPIS | 0.320 | 0.343 |
| **FastFusion-IF (ours)** | **0.339** | **0.404** |

FastFusion-IF beats GraphPPIS / RGCNPPIS / AGAT-PPIS on Test_315-28 (using AGAT-PPIS's own feature set),
and is the **best of all methods, including SOTA, on the unbound set**. Full per-metric tables are in
[`RESULTS.md`](RESULTS.md).

---

## Repository layout

```
fastfusion_if/        # the model + data pipeline (Python package)
scripts/              # command-line entry points (prepare data, build cache, train, evaluate)
configs/              # JSON experiment configs (model + training hyperparameters)
manifests/            # train/val/test split definitions (CSV) — defines the exact splits used
RESULTS.md            # full result tables (copy of the results summary)
requirements.txt
pyproject.toml
```

Caches, checkpoints, evaluation dumps and datasets are **not** stored in the repo (they are large and
regenerable); the steps below build them locally.

---

## 1. Requirements

**Hardware.** An NVIDIA GPU with a CUDA 12.1-capable driver. The paper used an RTX 3060 Ti (8 GB),
which is sufficient for the AGAT-PPIS benchmark. ~16–32 GB system RAM recommended. The full DIPS-Plus
training (Section 4) needs the same VRAM but considerably more disk and time.

**Software.** Linux, [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html), Python 3.10,
PyTorch 2.4.1 (CUDA 12.1). All other dependencies are in `requirements.txt`.

---

## 2. Installation

```bash
# clone your repository, then:
cd fastfusion_if_project

# 1) create and activate the environment
conda create -n fastfusion python=3.10 -y
conda activate fastfusion

# 2) install PyTorch 2.4.1 with CUDA 12.1 (the exact build used for the paper)
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
              pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3) install the remaining Python dependencies
pip install -r requirements.txt

# 4) install this project as a package (so `import fastfusion_if` works anywhere)
pip install -e .

# 5) (only if you will regenerate the DIPS-Plus splits in Section 4)
conda install -c bioconda -c conda-forge mmseqs2 -y
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.4.1 True NVIDIA GeForce RTX 3060 Ti   (or your GPU)
```

---

## 3. Reproduce the AGAT-PPIS benchmark result (main path)

This reproduces the headline numbers above. It needs only the (small) AGAT-PPIS benchmark data.

### 3.1 Get the benchmark data

Clone the official AGAT-PPIS repository, which provides the datasets, the `pdb/` structures, and the
precomputed evolutionary/structural features (PSSM, HMM, DSSP, resAF):

```bash
cd ~                      # or wherever you keep datasets
git clone https://github.com/AILBC/AGAT-PPIS.git
```

You should have:

```
AGAT-PPIS/Dataset/{Train_335.pkl, Test_60.pkl, Test_315-28.pkl, Test_315.pkl, UBtest_31-6.pkl}
AGAT-PPIS/Dataset/pdb/*.pdb
AGAT-PPIS/Feature/{pssm, hmm, dssp, resAF}/<protein_id>.npy
```

(Only `pssm`, `hmm`, `dssp`, `resAF` are used — 20+20+14+7 = 61 dims per residue. The distance-map /
psepos folders are not needed.)

### 3.2 Build the manifests (splits + labels + evolutionary features)

From the project root (`fastfusion_if_project`), with the `fastfusion` env active:

```bash
python scripts/prepare_benchmark.py \
  --dataset-dir ~/AGAT-PPIS/Dataset \
  --feature-dir ~/AGAT-PPIS/Feature \
  --config configs/full_v2_cached.json \
  --out-dir manifests/benchmark
```

This writes `manifests/benchmark/`: the split CSVs (`bench_train.csv`, `bench_test315.csv`,
`bench_ubtest.csv`, `bench_test60.csv`, `bench_all.csv`), the labels (`bench_labels.json`), and the
standardized evolutionary features (`bench_evo.npz` + `bench_evo_keys.json`). It should report
`evo features: 735 proteins, dim=61` and `train=335 val=60 test=287`.

### 3.3 Build the feature caches

Two caches. The first bakes geometry + ESM-2; the second attaches the evolutionary profiles by reusing
the first (so geometry/ESM are not recomputed).

```bash
# 3.3a  geometry + ESM-2 650M  (downloads ~2.5 GB of ESM weights on first run)
python scripts/precompute_cache.py \
  --manifest manifests/benchmark/bench_all.csv \
  --config configs/bench_esm_reg.json \
  --cache-dir cache/bench_esm \
  --splits all \
  --plm-model esm2_t33_650M_UR50D --plm-device cuda

# 3.3b  attach evolutionary profiles (fast; reuses geometry + ESM from 3.3a)
python scripts/precompute_cache.py \
  --manifest manifests/benchmark/bench_all.csv \
  --config configs/bench_evo_reg.json \
  --cache-dir cache/bench_evo \
  --splits all \
  --from-cache cache/bench_esm \
  --evo-file manifests/benchmark/bench_evo.npz
```

Sanity check that every cached chain has a uniform 61-dim evolutionary channel:

```bash
python - <<'PY'
import pickle, glob
dims=set(); n=0
for f in glob.glob("cache/bench_evo/**/*.pkl", recursive=True):
    for ex in pickle.load(open(f,'rb')):
        dims.add(int(ex.residue_features.shape[1])); n+=1
    if n>=400: break
print("residue_features widths:", sorted(dims), "(must be exactly [61])")
PY
```

### 3.4 Train (3 seeds)

The full model is **atom + surface + evolutionary** (`bench_evo_reg.json`). ESM is intentionally *off*
for the benchmark because it overfits at 335 training proteins. Each run auto-evaluates Test_315-28 at the
end and writes `best.pt`.

```bash
python scripts/train.py --manifest manifests/benchmark/bench_train.csv \
  --config configs/bench_evo_reg.json --out-dir runs/bench_evo_pp \
  --cache-dir cache/bench_evo --num-workers 4 --auto-resume

python scripts/train.py --manifest manifests/benchmark/bench_train.csv \
  --config configs/bench_evo_reg.json --out-dir runs/bench_evo_pp_s1 \
  --cache-dir cache/bench_evo --num-workers 4 --seed 1 --auto-resume

python scripts/train.py --manifest manifests/benchmark/bench_train.csv \
  --config configs/bench_evo_reg.json --out-dir runs/bench_evo_pp_s2 \
  --cache-dir cache/bench_evo --num-workers 4 --seed 2 --auto-resume
```

### 3.5 Evaluate the 3-seed ensemble

```bash
python scripts/evaluate_ensemble.py \
  --checkpoints runs/bench_evo_pp/best.pt runs/bench_evo_pp_s1/best.pt runs/bench_evo_pp_s2/best.pt \
  --manifest manifests/benchmark/bench_test315.csv --split test \
  --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_ens_test315

python scripts/evaluate_ensemble.py \
  --checkpoints runs/bench_evo_pp/best.pt runs/bench_evo_pp_s1/best.pt runs/bench_evo_pp_s2/best.pt \
  --manifest manifests/benchmark/bench_ubtest.csv --split test \
  --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_ens_ubtest
```

### 3.6 (Optional) Tune the decision threshold for F1 / MCC

AUPRC and AUROC are threshold-free, but F1/MCC depend on the operating threshold. Tune it on the
validation set (Test_60, which lives inside `bench_train.csv`) and apply to the test sets:

```bash
python scripts/evaluate_ensemble.py \
  --checkpoints runs/bench_evo_pp/best.pt runs/bench_evo_pp_s1/best.pt runs/bench_evo_pp_s2/best.pt \
  --manifest manifests/benchmark/bench_train.csv --split val \
  --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_ens_val
# read the "threshold" field of eval/bench_evo_ens_val/val_ensemble_metrics.json (= 0.63 in the paper), then:
python scripts/evaluate_ensemble.py \
  --checkpoints runs/bench_evo_pp/best.pt runs/bench_evo_pp_s1/best.pt runs/bench_evo_pp_s2/best.pt \
  --manifest manifests/benchmark/bench_test315.csv --split test --threshold 0.63 \
  --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_ens_test315_tuned
```

### 3.7 Expected results

| Test set | AUPRC | AUROC | MCC |
|---|---|---|---|
| Test_315-28 | 0.544 | 0.869 | 0.456 |
| UBtest_25 | 0.404 | 0.798 | 0.339 |

(Per-seed AUPRC on Test_315-28 is ~0.533 / 0.530 / 0.539; ensembling lifts it to 0.544. Small numerical
differences across machines/driver versions are normal.)

### Surface ablation (optional, demonstrates the surface contribution)

`configs/bench_evo_nosurf_reg.json` is identical to `bench_evo_reg.json` except `use_surface=false`.
Train it for 3 seeds on the same `cache/bench_evo` and compare: surface-on vs surface-off isolates the
contribution of the surface point cloud (+0.013 AUPRC on bound, +0.056 on unbound).

---

## 4. (Advanced) Train at scale on DIPS-Plus

This is the large-scale headline (PR-AUC 0.571 with ESM, a clean +0.064 ESM ablation on ~33,690
proteins). It requires downloading the large DIPS-Plus dataset and using MMseqs2 for leakage-safe splits.

```bash
# 4.1 download DIPS-Plus final/raw pairs (or fetch the archive directly)
python scripts/download_datasets.py --dataset dips-plus --out-dir data --zenodo-pattern final_raw_dips
#   archive: https://zenodo.org/record/8140981/files/final_raw_dips.tar.gz

# 4.2 build a leakage-safe split (MMseqs2 clustering at 30% identity)
python scripts/prepare_manifest.py \
  --data-dir data/DIPS-Plus/final/raw \
  --out manifests/dips_plus_mmseqs30.csv \
  --cluster-method mmseqs --identity 0.30 --threads 16

# 4.3 build the geometry + ESM-2 cache
python scripts/precompute_cache.py \
  --manifest manifests/dips_plus_mmseqs30.csv \
  --config configs/full_v2_esm_reg.json \
  --cache-dir cache/dips_plus_v2_esm --splits all \
  --plm-model esm2_t33_650M_UR50D --plm-device cuda

# 4.4 train
python scripts/train.py --manifest manifests/dips_plus_mmseqs30.csv \
  --config configs/full_v2_esm_reg.json --out-dir runs/full_dips_plus \
  --cache-dir cache/dips_plus_v2_esm --num-workers 8 --auto-resume

# 4.5 evaluate
python scripts/evaluate.py --checkpoint runs/full_dips_plus/best.pt \
  --manifest manifests/dips_plus_mmseqs30.csv --split test \
  --cache-dir cache/dips_plus_v2_esm --out-dir eval/full_dips_plus_test
```

The exact regularization recipe that unlocked ESM at scale (plm_dropout 0.35, weight_decay 0.03,
lr 1.5e-4, dropout 0.15) is encoded in the `*_esm_reg.json` configs; see `DATASETS.md` for more.

---

## 5. Configuration notes

Each run is driven by a JSON config in `configs/`. The benchmark-relevant ones:

| Config | Modalities | Cache |
|---|---|---|
| `bench_esm_reg.json` | atom + surface + ESM-2 | `cache/bench_esm` |
| `bench_evo_reg.json` | atom + surface + evolutionary (**full model**) | `cache/bench_evo` |
| `bench_evo_nosurf_reg.json` | atom + evolutionary (surface OFF; ablation) | `cache/bench_evo` |
| `bench_evo_esm_reg.json` | atom + surface + evolutionary + ESM | `cache/bench_evo` |

Toggle flags inside the config: `model.use_surface`, `model.use_residue_features` (evolutionary profiles),
`model.use_plm_features` (ESM). Caches are shared where the data block matches, so `bench_evo` is built
once and reused by the surface/evo/ESM variants.

---

## Citation

If you use this code, please cite the FastFusion-IF paper (in preparation) and the benchmark sources
(GraphPPIS, AGAT-PPIS, GTE-PPIS) and DIPS-Plus as appropriate.

## License

See `LICENSE`. (Add a license file before making the repository public — MIT is a common choice for
research code.)
