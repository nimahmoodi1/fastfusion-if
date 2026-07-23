# Released checkpoints

All values below were read directly from the checkpoint files, not from the paper.

| File | Config | Params | Trained with | Val PR-AUC | Own threshold |
|---|---|---|---|---|---|
| `fastfusion_if_full_seed42.pt` | `bench_evo_reg.json` | 1,388,133 | seed 42 | 0.5642 | 0.66 |
| `fastfusion_if_full_seed1.pt` | `bench_evo_reg.json` | 1,388,133 | seed 1 | 0.5749 | 0.64 |
| `fastfusion_if_full_seed2.pt` | `bench_evo_reg.json` | 1,388,133 | seed 2 | 0.5679 | 0.69 |
| `fastfusion_if_geometry_only.pt` | `bench_cached_reg.json` | 1,372,747 | seed 42 | — | — |

The three `full` checkpoints are the 3-seed ensemble reported in the paper. **Use the ensemble
threshold 0.63**, tuned once on the validation split for the averaged predictions — not the
per-model thresholds in the last column, which are each model's own best-F1 point.

## Which one do I want?

**`fastfusion_if_full_*`** — the paper model. Atom + mesh-free surface + 61-dim evolutionary and
structural profiles (PSSM 20 + HMM 20 + DSSP 14 + resAF 7). Reaches AUPRC 0.544 / MCC 0.456 on
Test_315-28 as a 3-seed ensemble.

**Requires the 61-dim profiles at inference time.** It will not run on a bare PDB. On the AGAT-PPIS
benchmark proteins the features ship with that repository; for a new protein you must generate
PSSM (PSI-BLAST), HMM (HHblits + UniClust30), DSSP and resAF yourself.

**`fastfusion_if_geometry_only.pt`** — atom + surface only, no profiles. AUPRC 0.395 / MCC 0.324
on Test_315-28 as a 3-seed ensemble (this file is the seed-42 member). Lower accuracy, but it runs
on any PDB with no external databases:

```bash
python scripts/infer_pdb.py \
  --checkpoint checkpoints/fastfusion_if_geometry_only.pt \
  --pdb my_protein.pdb --chain A
```

## What is inside a checkpoint

```python
{
  "model":               state_dict (274 tensors),
  "cfg":                 the full ExperimentConfig used for training,
  "surface_feature_dim": 4,
  "residue_feature_dim": 61   # 36 for the geometry-only model
  "plm_dim":             1280,
  "threshold":           this model's own best-F1 validation threshold,
  "best_val_metric":     "pr_auc",
  "best_val_score":      float,
}
```

Everything needed to rebuild the architecture is stored in the file, so `evaluate_ensemble.py`
and `infer_pdb.py` reconstruct the model without reading any config from disk.

To inspect one yourself:

```python
import torch
ck = torch.load("checkpoints/fastfusion_if_full_seed42.pt", map_location="cpu", weights_only=False)
n = sum(v.numel() for k, v in ck["model"].items() if not k.endswith("rbf_centers"))
print(f"{n:,} parameters")
print("residue_feature_dim:", ck["residue_feature_dim"])
print("seed:", ck["cfg"]["train"]["seed"])
```

## Training details

Trained on Train_335 from the AGAT-PPIS benchmark, validated on Test_60, for 40 epochs. AdamW,
lr 1.5e-4, weight decay 0.05, 3 warm-up epochs, gradient clipping 1.0, batch size 1 chain, mixed
precision, dropout 0.2. Loss is class-weighted BCE (positive weight estimated from the training
split, ≈5.2, so it varies slightly per seed) plus a soft-Dice term at weight 0.2. The checkpoint
kept is the epoch with the best validation PR-AUC.

Predictions are invariant to the orientation of the input structure by construction — every module
consumes only interatomic distances.
