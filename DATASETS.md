# Dataset notes

## First real run

Use DIPS-Plus final/raw files for training. Begin with a subset to verify the pipeline, then scale to the full dataset.

Example:

```bash
python scripts/download_datasets.py --dataset dips-plus --out-dir data --zenodo-pattern final_raw_dips
python scripts/prepare_manifest.py --data-dir data/DIPS-Plus/final/raw --out manifests/dips_plus_mmseqs30.csv --cluster-method mmseqs --identity 0.30 --threads 16
python scripts/train.py --manifest manifests/dips_plus_mmseqs30.csv --config configs/default.json --out-dir runs/fastfusion_if_dips_plus
```

## External benchmark

Use DB5/DB5-Plus as an external test after the model is selected.

```bash
python scripts/download_datasets.py --dataset db5-plus --out-dir data --extract
python scripts/prepare_manifest.py --data-dir data/DB5/final/raw --out manifests/db5_mmseqs30.csv --cluster-method mmseqs --identity 0.30 --threads 16
python scripts/evaluate.py --checkpoint runs/fastfusion_if_dips_plus/best.pt --manifest manifests/db5_mmseqs30.csv --split test --out-dir eval/db5_external
```

Because DB5 is small, you may also create a one-split manifest with all rows marked `test`:

```bash
python scripts/make_test_manifest.py --data-dir data/DB5/final/raw --out manifests/db5_all_test.csv
python scripts/evaluate.py --checkpoint runs/fastfusion_if_dips_plus/best.pt --manifest manifests/db5_all_test.csv --split test --out-dir eval/db5_external
```

## PINDER

PINDER is valuable but large. Use it after the model is stable. Prefer PINDER official splits/evaluation harness if your experiment is meant to compare with docking/general PPI models.
