# FastFusion-IF — phase plan (DIPS-Plus first, then external benchmarks)

Goal: a residue-level protein–protein interface predictor that is competitive
with and ideally beats published methods, with a fair, reviewer-proof comparison
and a clean ablation story. We lock down DIPS-Plus first, then expand.

Hardware reality: RTX 3060 Ti (8 GB) now, possibly more later. Everything below
fits in 8 GB. Capacity scaling notes are at the end for when you upgrade.

Decision rule used throughout: develop on the LARGE split; only promote a change
to the FULL split if it improves PR-AUC or MCC on large. Keep v2-full (PR-AUC
0.5127, MCC 0.4526) as the number to beat.

------------------------------------------------------------------------------
PART 1 — DIPS-Plus (do all of this before touching other datasets)
------------------------------------------------------------------------------

Phase 0 — Sanity + cache (½ day, mostly waiting)
  * Run scripts/smoke_test_upgrades.py (seconds) — confirms the new code wiring.
  * Build the example cache for the LARGE split, then FULL:
      python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30_large.csv \
          --config configs/large_v2_cached.json --cache-dir cache/dips_plus_v2 --splits train,val,test
  * This removes the per-epoch surface/graph/label recomputation that capped you
    at 12 epochs and caused the loader segfaults.

Phase 1 — Properly-trained v2 baseline (the honest re-baseline)   [GATE]
  * Train v2 from the cache for ~30 epochs on large, then ~40 on full:
      python scripts/train.py --manifest manifests/dips_plus_mmseqs30_large.csv \
          --config configs/large_v2_cached.json --out-dir runs/large_v2_cached --num-workers 4
  * Expectation: equal or better than current v2 because it is no longer
    undertrained (your val PR-AUC was still rising at epoch 10/12).
  * GATE: record the new large/full numbers. This is your reference point; every
    later change is measured against it.

Phase 2 — ESM-2 features (biggest expected lever)                 [GATE]
  * Build a PLM cache (separate dir!). Reuse the geometry you already cached so
    only the ESM embeddings are computed:
      python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30_large.csv \
          --config configs/large_v2_cached.json --cache-dir cache/dips_plus_v2_esm \
          --from-cache cache/dips_plus_v2 --splits train,val,test --plm-model esm2_t33_650M_UR50D
      python scripts/train.py --manifest manifests/dips_plus_mmseqs30_large.csv \
          --config configs/large_v2_esm.json --out-dir runs/large_v2_esm \
          --cache-dir cache/dips_plus_v2_esm --num-workers 4 --auto-resume
  * If 650M is tight on disk/time, use esm2_t30_150M_UR50D (640-d) — still strong.
  * Injection: the ESM features are concatenated into the residue token before
    the residue-context transformer (configs use plm_inject="concat"), and the
    injection is ZERO-INITIALISED, so the model starts identical to the no-PLM
    baseline and learns to use ESM — it can't degrade the geometric backbone at
    init. This is the corrected version of the failed v3-lite experiment (which
    used a weak 0.25-scaled additive path plus harmful hand-crafted features).
  * GATE: ESM should give the largest single jump. If it does, promote to full
    (configs/full_v2_esm.json). This becomes "FastFusion-IF-v2+".

Phase 3 — Richer surface features (strengthens the core claim)
  * Your headline is "molecular surface representation" but the descriptor is 4
    thin scalars. The rich set adds burial, PCA shape, H-bond donor/acceptor, etc.
      python scripts/precompute_cache.py --manifest manifests/dips_plus_mmseqs30_large.csv \
          --config configs/large_v2_rich_surface.json --cache-dir cache/dips_plus_v2_rich --splits train,val,test
      python scripts/train.py --manifest manifests/dips_plus_mmseqs30_large.csv \
          --config configs/large_v2_rich_surface.json --out-dir runs/large_v2_rich --num-workers 4
  * Run this as a clean ablation (with and without), so the paper can attribute
    the gain to the surface representation specifically.

Phase 4 — Loss / calibration polish
  * Try focal and Tversky on large (configs/large_v2_focal.json). Keep only if it
    improves PR-AUC/MCC. Report ROC-AUC/PR-AUC (threshold-free) as headline, plus
    F1/MCC at the validation-selected threshold and at 0.5, plus ECE (now logged).
  * Do this AFTER the architecture is settled so you change one thing at a time.

Phase 4b — Final squeeze: TTA + ensembling (cheap SOTA-chasing)
  * Test-time augmentation: re-evaluate the best model averaging probabilities
    over several random rotations (the model is ~rotation-invariant, so this
    lowers variance and nudges PR-AUC/MCC/ECE up at zero training cost):
      python scripts/evaluate.py --checkpoint runs/full_v2_esm/best.pt \
          --manifest manifests/dips_plus_mmseqs30.csv --split test \
          --cache-dir cache/dips_plus_v2_esm --tta 8 --out-dir eval/full_v2_esm_tta
  * Ensemble: average a few independently-trained models (different seeds, or
    geometry-only + rich-surface + ESM-2). This almost always beats any single
    member and is how you claim the final headline number:
      python scripts/evaluate_ensemble.py \
          --checkpoints runs/full_v2_esm/best.pt runs/full_v2_rich/best.pt runs/full_v2_cached/best.pt \
          --manifest manifests/dips_plus_mmseqs30.csv --split test \
          --cache-dir cache/dips_plus_v2_esm --tta 8 --out-dir eval/ensemble_full
  * Report BOTH the single best model and the ensemble (ensembles are standard
    and accepted, but reviewers also want the single-model number).

Phase 5 — Fair same-split baselines on DIPS-Plus                  [PUBLISHABILITY]
  * Run pretrained PeSTo (github.com/LBM-EPFL/PeSTo, model i_v4_1) and ScanNet
    (github.com/jertubiana/ScanNet, no-MSA first) on YOUR DIPS-Plus test chains.
  * Map their per-residue scores to your 5 Å heavy-atom label definition and
    compute the same metrics. This turns the DIPS-Plus comparison into a valid,
    same-split one. (I can write this evaluation glue when you reach this phase.)

End-of-Part-1 deliverable: ablation ladder on DIPS-Plus
  Geo (atom+surface+fusion) -> +resctx (v2) -> +rich surface -> +ESM-2,
  each with global + median-per-protein ROC-AUC/PR-AUC/F1/MCC/ECE, plus PeSTo and
  ScanNet on the same test set. That is a publishable results core on its own.

------------------------------------------------------------------------------
PART 2 — External datasets (only after Part 1 numbers are solid)
------------------------------------------------------------------------------

Phase 6 — Standard PPIS benchmark (makes you comparable to the cited methods)
  * Use the AGAT-PPIS release (github.com/AILBC/AGAT-PPIS): Train_335 / Test_60 /
    Test_315-28 / UBtest_31-6, with labels + PDBs included. Train FastFusion on
    Train_335 and report the three test sets. These numbers are DIRECTLY
    comparable to GraphPPIS / AGAT-PPIS / GACT-PPIS / EquiPPIS / GTE-PPIS / etc.
  * Tiny (335 train chains) -> minutes to a couple of hours on a 3060 Ti.
  * I can write the Train_335 .pkl loader + a config so this is turnkey.

Phase 7 — Cross-dataset generalisation
  * DB5.5 and a PINDER leakage-aware test subset, evaluated with your DIPS-Plus
    or Train_335 trained model, for a generalisation table.

------------------------------------------------------------------------------
Capacity scaling (when you get a bigger GPU)
------------------------------------------------------------------------------
  * Raise atom_dim/surface_dim/fusion_dim 96 -> 128/192, n_atom_layers 4 -> 6,
    n_fusion_layers 2 -> 3, n_residue_layers 2 -> 3. Your current ~1.3M-param
    model is small; more capacity + ESM should close the gap to the best methods.
  * Raise max_surface_points 2048 -> 4096 and n_surface_dirs 16 -> 24 for a denser
    surface once memory allows.
  * batch_size can stay 1 with graph batching; gradient accumulation is an easy
    add if you want a larger effective batch.

What to ask me to build next, in priority order:
  1. (Phase 5) the PeSTo/ScanNet -> DIPS-Plus same-split evaluation glue.
  2. (Phase 6) the Train_335 benchmark loader + config (turnkey Track A).
  3. The paper methods section + results-table skeleton from your real numbers.
