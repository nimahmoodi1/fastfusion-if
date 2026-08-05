# XAI — every command, in order

Two directories are involved:

| Path | Role |
|---|---|
| `~/Nima/fastfusion_if_project_real_splits_v2/fastfusion_if_project` | **main project** — checkpoints, caches, datasets |
| `~/fastfusion-if-update` | **repo clone** — where code lives and what gets pushed |

Run the code from the repo clone and point it at the main project for **both
the data and the manifests**.

Use `$PROJ/manifests/...`, not the repo clone's copies. The committed manifests
were sanitised to relative paths to keep your username out of the public
repository, and the feature cache is keyed by a hash of the *full path string* —
so relative and absolute forms never match the same cache file. (If you do want
the committed manifests, pass `--pdb-root ~/`.)

Set these once per shell:

```bash
cd ~/fastfusion-if-update
conda activate fastfusion
export PROJ=~/Nima/fastfusion_if_project_real_splits_v2/fastfusion_if_project
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1     # your ROS 2 install breaks pytest otherwise
```

---

## Step 0 — install the fixed files

```bash
git switch -c feature/xai        # skip if the branch already exists

# from the unzipped package
cp -r xai/fastfusion_if/xai        fastfusion_if/
cp    xai/scripts/run_xai.py       scripts/
cp    xai/tests/test_xai.py        tests/
cp    xai/tests/conftest.py        tests/          # NEW — fixes the 10 fixture errors
cp    xai/pytest.ini               .               # NEW — fixes the ROS plugin crash
cp    xai/configs/xai_default.yaml configs/
cp -r xai/paper                    .
cp    xai/README_XAI.md xai/requirements-xai.txt xai/COMMANDS.md xai/FIXES.md .

pip install -r requirements-xai.txt
```

## Step 1 — tests (no GPU)

```bash
pytest tests/test_xai.py -q
# expect: 18 passed, 10 skipped
```

Both `pytest` and `python -m pytest` now work; previously only the latter did.
Skipped, not errored, is the correct result — the model tier needs a checkpoint.
To run all 28:

```bash
pytest tests/test_xai.py -q \
  --ckpt $PROJ/runs/bench_evo_pp/best.pt \
  --manifest $PROJ/manifests/benchmark/bench_test315.csv \
  --cache-dir $PROJ/cache/bench_evo
```

If you would rather use the repo clone's manifests, add `--pdb-root ~/`.

## Step 2 — inspect the real batch  ← **run this before anything else**

```bash
python scripts/run_xai.py keys \
  --checkpoint $PROJ/runs/bench_evo_pp/best.pt \
  --manifest $PROJ/manifests/benchmark/bench_test315.csv --split test \
  --cache-dir $PROJ/cache/bench_evo
```

Check three lines of the output:

- `label key : '...'` — if it says NOT FOUND, add the real name to `_LABEL_KEYS`
  near the top of `scripts/run_xai.py`. Nothing else needs changing.
- `required model inputs present : YES`
- `label count matches n_residues: YES` — if NO, the dataset yields more than one
  `ChainExample` per item and attribution rows would not line up with residues.
  Send me the output and stop here.

## Step 3 — smoke test

```bash
python scripts/run_xai.py smoke \
  --checkpoint $PROJ/runs/bench_evo_pp/best.pt \
  --manifest $PROJ/manifests/benchmark/bench_test315.csv --split test \
  --cache-dir $PROJ/cache/bench_evo --config configs/xai_default.yaml
```

Five checks now print, plus the model's measured run-to-run noise:

```
model run-to-run noise (atomicAdd)     : 3.8e-06     <- expected, not a fault
baseline identical with hooks installed : PASS
gate decomposition identity            : PASS
atom + surface == geom                 : PASS
IG completeness, global (64 steps)     : 0.0041  PASS
IG completeness, per-residue (self/mean): 0.0123  PASS
  [diagnostic] same with a zero baseline: 0.2199  (expected to be poor)
```

The zero-baseline diagnostic line is *expected* to be poor and is not a failure:
the classifier head begins with a scale-invariant LayerNorm, so an IG path from
the origin never leaves a level set of the head. That is why `scope="self"`
defaults to a mean baseline. `scope="total"`, which every reported statistic
uses, is unaffected.

The nonzero noise line is normal: CUDA scatter reductions use `atomicAdd`, so
this model never reproduces bitwise. Every check below it is asserted against
that measured spread rather than against exact equality.

**If `gate decomposition identity` fails, stop** — the decoder differs from the
released source and the atom/surface split would be invalid.

**If the global completeness line fails**, raise `ig_steps` in
`configs/xai_default.yaml` from 64 to 128 and re-run before going further. The
`self/mean` line failing matters only if you plan single-residue case studies.

## Step 4 — attributions (the main run, ~1 min per set)

```bash
for SET in test315 ubtest btest test60; do
  [ -f $PROJ/manifests/benchmark/bench_${SET}.csv ] || { echo "skip $SET"; continue; }
  python scripts/run_xai.py attribute \
    --checkpoint $PROJ/runs/bench_evo_pp/best.pt \
    --manifest $PROJ/manifests/benchmark/bench_${SET}.csv --split test \
    --cache-dir $PROJ/cache/bench_evo --config configs/xai_default.yaml \
    --out-dir results/xai/${SET}
done
```

Resumable — re-running skips proteins that already have **both** their
attribution CSV and their gates `.npz`. (Checking only the CSV would
permanently skip a protein whose CSV was written before a later step failed.) Failures land in
`results/xai/<set>/failed.csv` and do not stop the run.

Each run also writes `attribution_quality.csv` and prints a line like
`IG completeness: median 0.0038, 0/287 proteins above the 0.02 threshold`. If
proteins are listed above the threshold, their attributions have not converged;
raise `ig_steps` and re-run just those with `--overwrite`.

For a single-residue case study, add `--scope self` — that gives exact
per-residue completeness by bypassing the residue-context encoder. Use the
default `--scope total` for everything aggregated over a protein.

Check both afterwards:

```bash
for SET in test315 ubtest btest test60; do
  n=$(ls results/xai/$SET/residue_attributions/*.csv 2>/dev/null | wc -l)
  f=$( [ -f results/xai/$SET/failed.csv ] && tail -n +2 results/xai/$SET/failed.csv | wc -l || echo 0)
  echo "$SET: $n succeeded, $f failed"
  [ -f results/xai/$SET/attribution_quality.csv ] && \
    python -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); \
print(f'   completeness median {d.ig_global_rel_error.median():.4f}, \
{(d.ig_global_rel_error>0.02).sum()} over threshold')" results/xai/$SET/attribution_quality.csv
done
```

## Step 5 — interventions

```bash
for SET in test315 ubtest; do
  python scripts/run_xai.py ablate \
    --checkpoint $PROJ/runs/bench_evo_pp/best.pt \
    --manifest $PROJ/manifests/benchmark/bench_${SET}.csv --split test \
    --cache-dir $PROJ/cache/bench_evo --out-dir results/xai/${SET}
done
```

## Step 6 — the honest surface estimate

Test-time surface zeroing is out of distribution. The retrained surface-off
checkpoint is the comparison to believe:

```bash
python scripts/run_xai.py attribute \
  --checkpoint $PROJ/runs/bench_evo_nosurf_pp/best.pt \
  --manifest $PROJ/manifests/benchmark/bench_ubtest.csv --split test \
  --cache-dir $PROJ/cache/bench_evo --out-dir results/xai/ubtest_nosurf
```

## Step 7 — faithfulness

```bash
for SET in test315 ubtest; do
  python scripts/run_xai.py faithful \
    --checkpoint $PROJ/runs/bench_evo_pp/best.pt \
    --manifest $PROJ/manifests/benchmark/bench_${SET}.csv --split test \
    --cache-dir $PROJ/cache/bench_evo --out-dir results/xai/${SET} \
    --score-col attr_evolutionary
done
```

If `faithfulness gap mean` is at or below zero, the attributions carry no more
information than chance and Steps 4–6 must not be reported. That is a real
possible outcome, not a bug.

## Step 8 — statistics and tables (no GPU)

```bash
for SET in test315 ubtest btest test60; do
  [ -d results/xai/$SET/residue_attributions ] && \
    python scripts/run_xai.py analyse --out-dir results/xai/${SET}
done
```

## Step 9 — send me the results

```bash
cd ~/fastfusion-if-update
zip -r ~/xai_results.zip \
  results/xai/*/tables results/xai/*/error_analysis \
  results/xai/*/faithfulness results/xai/*/modality_ablation \
  results/xai/*/failed.csv 2>/dev/null
```

That is small (tables only, no raw tensors). I will write the Results section
from the real numbers.

## Step 10 — commit and push, after the numbers exist

```bash
git add fastfusion_if/xai scripts/run_xai.py tests/test_xai.py tests/conftest.py \
        pytest.ini configs/xai_default.yaml paper/xai \
        README_XAI.md requirements-xai.txt COMMANDS.md FIXES.md .gitignore
git status --short          # confirm no results/xai/raw or *.npz
git commit -m "Add optional XAI module: gate analysis, injection-point IG, interventions

Instrumentation is by forward hook only; models/model.py is unmodified and
baseline predictions are bit-identical with XAI disabled.

Attribution is computed at the three residue injection points, which are
additive at a common point in a common space and therefore comparable across
modalities. The geometric term splits exactly by the learned gate. Gate values
are reported but validated against attribution and intervention rather than
treated as modality reliance, because the gate operates on post-fusion
representations that are already entangled."
git push -u origin feature/xai
gh pr create --draft --title "XAI module for FastFusion-IF" --body-file FIXES.md
```

Do not merge to `master` until the numbers are in the paper.
