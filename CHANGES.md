# CHANGES — FastFusion-IF upgrade (DIPS-Plus phase)

This documents exactly what changed versus your uploaded project, how to run it,
and what was vs was not validated in the environment that produced these edits.

## Validation status (read this)

Validated here (Python without torch): all configs load + round-trip; the rich
surface feature path produces correct shapes/finite values; the basic path stays
4-dim; ECE works; the augmentation rotation is a proper rotation that preserves
pairwise distances (so cached graphs stay valid); a ChainExample with PLM
embeddings pickles and round-trips; every edited file byte-compiles.

NOT runnable here (no GPU / no torch / no dataset / no network): the torch model
forward/backward, the training loop, ESM-2 download, and anything reading .dill.
Before any real run, execute the smoke test, which exercises the model + loss +
collate + PLM wiring on synthetic data:

    python scripts/smoke_test_upgrades.py

Then a 1-epoch debug run through the cached path:

    python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30_debug.csv \
        --config configs/debug_v2_resctx.json --cache-dir cache/debug --splits train,val,test
    python scripts/train.py --manifest manifests/dips_plus_mmseqs30_debug.csv \
        --config configs/debug_v2_resctx.json --out-dir runs/debug_cached \
        --cache-dir cache/debug --epochs 1 --num-workers 2

Everything new is behind config flags; with default configs the pipeline behaves
exactly as your current v2.

## New files

* `fastfusion_if/data/cached_dataset.py` — `CachedInterfaceDataset`: loads
  precomputed examples; applies fresh per-epoch rotation/jitter at load.
* `fastfusion_if/data/surface_features_rich.py` — 10-d surface descriptor
  (adds burial/atom-density, PCA shape planarity & curvedness, H-bond
  donor/acceptor, aromatic context to your 4 originals). Gated by
  `data.surface_feature_set == "rich"`.
* `fastfusion_if/data/plm.py` — `ESM2Extractor`: lazy-imported ESM-2 wrapper with
  per-sequence disk caching; builds embeddings from residue_names so rows align
  with residues exactly.
* `scripts/precompute_cache.py` — one-time cache builder; `--plm-model` bakes in
  ESM-2 embeddings; atomic writes; resumable; shardable with `--shard i/N`.
  `--from-cache <dir>` reuses geometry from an existing (non-PLM) cache and only
  computes the ESM embeddings, which avoids recomputing the surface/graphs/labels
  when you already have a baseline cache (big time saver, especially on full).
* `scripts/smoke_test_upgrades.py` — fast wiring test on synthetic data.
* New configs: `large_v2_cached`, `full_v2_cached`, `large_v2_rich_surface`,
  `large_v2_esm`, `full_v2_esm`, `large_v2_focal`, `large_v2_esm_rich`.
* `PHASE_PLAN.md` — the sequenced plan and decision gates.

## Edited files (all backward compatible)

* `fastfusion_if/config.py`
  - DataConfig: `surface_feature_set="basic"`, `burial_radius=10.0`,
    `shape_k_neighbors=16`, `cache_dir=None`.
  - ModelConfig: `use_plm_features=False`, `plm_dim=0`, `plm_dropout=0.10`,
    `plm_inject="concat"` ("concat" or "add").
  - TrainConfig: `focal_weight=0.0`, `focal_gamma`, `focal_alpha`,
    `tversky_weight=0.0`, `tversky_alpha/beta/gamma`. Defaults reproduce the
    current weighted-BCE + Dice loss exactly.
* `fastfusion_if/losses.py` — adds numerically-stable focal BCE and
  (focal-)Tversky terms; `interface_loss` superset, default behaviour unchanged.
* `fastfusion_if/metrics.py` — adds `expected_calibration_error`; `binary_metrics`
  now also returns `ece`.
* `fastfusion_if/data/structures.py` — `ChainExample.residue_plm: Optional[...] = None`.
* `fastfusion_if/data/surface.py` — dispatches to the rich descriptor when
  configured; basic path untouched.
* `fastfusion_if/data/collate.py` — emits `batch["residue_plm"]` only when all
  examples carry aligned embeddings; else `None`.
* `fastfusion_if/data/dataset.py` — augmentation now uses fresh RNG each call
  (fixes the fixed-per-file rotation bug) and rotates residue_pos consistently.
* `fastfusion_if/models/model.py` — ESM-2 features are projected (LayerNorm on
  the raw embeddings for scale control) and injected into the residue token
  BEFORE the residue-context transformer. Default `plm_inject="concat"` mixes
  `LayerNorm([residue_h | plm_h])` through a Linear; `"add"` uses a small
  non-zero learnable gate. The injection is IMMEDIATELY ACTIVE (receives gradient
  from step 1).
  NOTE — fixed a dead-path bug from the previous build: the PLM projection AND the
  concat-combine were BOTH zero-initialised, which compounded into a saddle where
  the entire ESM pathway received exactly zero gradient and never trained (its
  curve overlaid the no-ESM baseline). The current injection has no such dead path.
* `scripts/check_plm_cache.py` (NEW) — inspects ESM embeddings baked into a cache
  and reports presence, alignment, finiteness, norm distribution, and a
  contextuality test (same amino acid at different positions must have different
  vectors). Use it to confirm ESM embeddings are real and contextual, separately
  from training.
* `fastfusion_if/evaluation.py` — adds `collect_predictions_tta(...)`:
  rotation-based test-time augmentation that averages probabilities over N
  orientations (first pass = identity). Only positions are rotated; scalar
  surface descriptors are rotation-invariant.
* `scripts/train.py` — `--cache-dir/--num-workers/--epochs` overrides; cached or
  on-the-fly dataset selection; computes plm_dim from the batch; clear error if
  `use_plm_features` but no PLM in data; passes new loss params; saves `plm_dim`
  in best.pt/last.pt.
* `scripts/evaluate.py` — `--cache-dir`; passes `plm_dim` from the checkpoint;
  `--tta N` averages probabilities over N test-time rotations (1 = off).
* `scripts/evaluate_ensemble.py` (NEW) — averages per-residue probabilities
  across several checkpoints (optionally with `--tta` and per-model `--weights`),
  then computes the standard metrics once on the averaged probabilities. Members
  may differ in architecture/features; they must share the manifest split.
* `scripts/infer_pdb.py` — passes `plm_dim` from the checkpoint.

## Quick command reference (DIPS-Plus)

Baseline (Phase 1):
    python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30_large.csv \
        --config configs/large_v2_cached.json --cache-dir cache/dips_plus_v2 --splits train,val,test
    python scripts/train.py --manifest manifests/dips_plus_mmseqs30_large.csv \
        --config configs/large_v2_cached.json --out-dir runs/large_v2_cached --num-workers 4 \
        2>&1 | tee runs/large_v2_cached/train.log

ESM-2 (Phase 2):
    python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30_large.csv \
        --config configs/large_v2_cached.json --cache-dir cache/dips_plus_v2_esm \
        --splits train,val,test --plm-model esm2_t33_650M_UR50D
    python scripts/train.py --manifest manifests/dips_plus_mmseqs30_large.csv \
        --config configs/large_v2_esm.json --out-dir runs/large_v2_esm --num-workers 4

Rich surface (Phase 3):
    python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30_large.csv \
        --config configs/large_v2_rich_surface.json --cache-dir cache/dips_plus_v2_rich --splits train,val,test
    python scripts/train.py --manifest manifests/dips_plus_mmseqs30_large.csv \
        --config configs/large_v2_rich_surface.json --out-dir runs/large_v2_rich --num-workers 4

Evaluate any checkpoint (use the SAME cache the checkpoint was trained with):
    python scripts/evaluate.py --checkpoint runs/large_v2_esm/best.pt \
        --manifest manifests/dips_plus_mmseqs30_large.csv --split test \
        --cache-dir cache/dips_plus_v2_esm --out-dir eval/large_v2_esm_test

Final numbers with TTA + ensembling (Phase 4b):
    python scripts/evaluate.py --checkpoint runs/full_v2_esm/best.pt \
        --manifest manifests/dips_plus_mmseqs30.csv --split test \
        --cache-dir cache/dips_plus_v2_esm --tta 8 --out-dir eval/full_v2_esm_tta
    python scripts/evaluate_ensemble.py \
        --checkpoints runs/full_v2_esm/best.pt runs/full_v2_rich/best.pt runs/full_v2_cached/best.pt \
        --manifest manifests/dips_plus_mmseqs30.csv --split test \
        --cache-dir cache/dips_plus_v2_esm --tta 8 --out-dir eval/ensemble_full

## Notes / gotchas

* Use a DIFFERENT `--cache-dir` per feature variant (basic / rich / esm). The new
  configs already point to distinct dirs. PLM rebuilds are guarded: if you point a
  PLM build at an existing non-PLM cache it will rebuild those files with PLM.
* ESM-2 needs `pip install fair-esm`. Disk: ~1280 floats/residue (fp16 cached).
* Keep `2>&1 | tee -a runs/NAME/train.log` when resuming (append), as before.
* You can raise `--num-workers` now that the loader no longer runs heavy native
  code; start at 4 and watch RAM.
* Deprecation cleanup (cosmetic, behaviour unchanged): AMP now uses
  `torch.amp.GradScaler("cuda", ...)` and all `torch.load` calls pass
  `weights_only=False` explicitly (our checkpoints are trusted and contain the
  config dict). This silences the FutureWarnings you saw in the logs.

## Publishability: same-split external comparison (PeSTo / ScanNet)
* `scripts/export_test_pdbs.py` — exports the evaluated test chains to single-chain
  PDBs (+ `pdb_index.csv`) so external predictors run on exactly our residues.
* `scripts/adapt_bfactor_to_scores.py` — converts B-factor-scored PDBs (PeSTo output)
  into a per-residue scores CSV (max/mean over a residue's atoms).
* `scripts/compare_external.py` — inner-joins an external scores CSV with our
  per-residue prediction CSV on (group_id, chain, res_seq, insertion) and computes
  identical metrics for BOTH methods on the intersection, reporting coverage. This is
  the reviewer-proof way to compare against published methods on our own split.

## Route B: run FastFusion-IF on the AGAT-PPIS / GraphPPIS benchmark (article-comparable)
* `scripts/prepare_benchmark.py` — reads the benchmark .pkl files ({id:[seq,labels]})
  and Dataset/pdb/*.pdb, aligns each protein's per-residue 0/1 labels onto our parsed
  residues (exact when lengths match, else via longest-matching-blocks), and writes
  bench_labels.json + manifests (bench_train/test60/test315/ubtest/all) + an alignment
  report. Train_335 is split into train/val; Test_60/Test_315-28/UBtest_31-6 are test.
* `configs/bench_esm_reg.json` / `configs/bench_cached_reg.json` — small-data recipe
  (stronger reg: plm_dropout 0.4, dropout 0.2, wd 0.05) for the 335-protein train set.
* Uses the existing `precompute_cache.py --labels-file` + `make_chain_example(label_override=...)`
  path (their labels, not our 5A rule; single-chain PDBs kept).
