# FastFusion-IF — explainability (XAI)

Optional module that answers: *why does FastFusion-IF call this residue an
interface residue, and how much does that call depend on atomic geometry, the
molecular surface, and evolutionary profiles?*

It does not change training or inference. `models/model.py` is untouched;
everything works through forward hooks, so checkpoints stay compatible and
predictions with XAI disabled are bit-identical to the baseline.

## Install

    pip install -r requirements-xai.txt   # captum-free: only numpy/scipy/pandas/pyyaml/matplotlib

## Quick start

    # 1. verify the instrumentation on one protein (fast, CPU is fine)
    python scripts/run_xai.py smoke \
      --checkpoint runs/bench_evo_pp/best.pt \
      --manifest manifests/benchmark/bench_test315.csv --split test \
      --cache-dir cache/bench_evo

    # 2. attributions over a split
    python scripts/run_xai.py attribute \
      --checkpoint runs/bench_evo_pp/best.pt \
      --manifest manifests/benchmark/bench_test315.csv --split test \
      --cache-dir cache/bench_evo --out-dir results/xai

    # 3. statistics (no GPU, no torch needed)
    python scripts/run_xai.py analyse --out-dir results/xai

## What it computes

| Output | Meaning |
|---|---|
| `attr_atom`, `attr_surface` | Integrated-Gradients attribution split exactly by the learned gate |
| `attr_evolutionary` | attribution of the `λ·φ_r(f)` term |
| `reliance_*`, `srs` | normalised modality shares; `srs` is the Surface Reliance Score |
| `gate_mean` | mean of the learned gate — reported, **not** treated as reliance |
| `ig_convergence_delta` | IG completeness residual, so you can see how much is unexplained |

## How to read the results

Attribution, gate values and interventions answer different questions and are
allowed to disagree. Read them together:

* **Gate** — which pathway is weighted. It operates on post-fusion
  representations, so it does *not* tell you where information came from.
* **Attribution** — which term the gradient flows through.
* **Intervention** — what actually happens when a modality is removed.

Where they disagree, believe the intervention, and prefer the retrained
surface-off comparison (`runs/bench_evo_nosurf_pp*`) over test-time surface
zeroing, because the retrained model was free to compensate.

## Limitations

Post-hoc attribution describes the model, not the biology. High surface reliance
means the surface pathway drives that prediction; it does not mean the residue
binds through surface complementarity. See `paper/xai/limitations_xai.md`.

## Tests

    pytest tests/test_xai.py -q

The statistical tier runs without torch. The model tier is skipped when torch is
absent, so CI without a GPU still exercises everything that does not need one.
