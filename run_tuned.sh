set -e
cd ~/Nima/fastfusion_if_project_real_splits_v2/fastfusion_if_project

CKPT_ON="runs/bench_evo_pp/best.pt runs/bench_evo_pp_s1/best.pt runs/bench_evo_pp_s2/best.pt"
CKPT_OFF="runs/bench_evo_nosurf_pp/best.pt runs/bench_evo_nosurf_pp_s1/best.pt runs/bench_evo_nosurf_pp_s2/best.pt"

echo "===== SURFACE-ON: tune threshold on val (Test_60, inside bench_train.csv) ====="
python scripts/evaluate_ensemble.py --checkpoints $CKPT_ON \
    --manifest manifests/benchmark/bench_train.csv --split val \
    --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_ens_val
T_ON=$(python -c "import json;print(json.load(open('eval/bench_evo_ens_val/val_ensemble_metrics.json'))['threshold'])")
echo ">>> tuned threshold (surface-on) = $T_ON"
python scripts/evaluate_ensemble.py --checkpoints $CKPT_ON \
    --manifest manifests/benchmark/bench_test315.csv --split test --threshold "$T_ON" \
    --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_ens_test315_tuned

echo "===== SURFACE-OFF: tune threshold on val ====="
python scripts/evaluate_ensemble.py --checkpoints $CKPT_OFF \
    --manifest manifests/benchmark/bench_train.csv --split val \
    --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_nosurf_ens_val
T_OFF=$(python -c "import json;print(json.load(open('eval/bench_evo_nosurf_ens_val/val_ensemble_metrics.json'))['threshold'])")
echo ">>> tuned threshold (surface-off) = $T_OFF"
python scripts/evaluate_ensemble.py --checkpoints $CKPT_OFF \
    --manifest manifests/benchmark/bench_test315.csv --split test --threshold "$T_OFF" \
    --cache-dir cache/bench_evo --tta 8 --out-dir eval/bench_evo_nosurf_ens_test315_tuned

echo
echo "===== TUNED RESULTS (Test_315-28 global, at val-tuned threshold) ====="
python - <<'PY'
import json
for name,path in [("surface-ON  (full)","eval/bench_evo_ens_test315_tuned/test_ensemble_metrics.json"),
                  ("surface-OFF       ","eval/bench_evo_nosurf_ens_test315_tuned/test_ensemble_metrics.json")]:
    g=json.load(open(path)); m=g["global"]
    print(f"{name}  thr={g['threshold']:.2f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}  AUPRC={m['pr_auc']:.4f}  ROC={m['roc_auc']:.4f}")
PY
