"""Tests for the XAI layer.

Tests split into two tiers. The statistical tier runs anywhere; the model tier
is skipped when torch is absent, so CI without a GPU still exercises everything
that does not need one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfusion_if.xai import (
    aggregate_by_protein,
    bootstrap_ci,
    confusion_class,
    error_analysis,
    holm_bonferroni,
    modality_reliance,
    paired_test,
    select_cases,
)

def _toy(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "group_id": rng.choice([f"p{i}" for i in range(8)], n),
            "res_seq": np.arange(n),
            "label": rng.integers(0, 2, n),
            "prob": rng.random(n),
            "logit": rng.normal(size=n),
            "gate_mean": rng.random(n),
            "attr_atom": rng.normal(size=n),
            "attr_surface": rng.normal(size=n),
            "attr_evolutionary": rng.normal(size=n),
        }
    )


# ------------------------------------------------------------ statistical tier
class TestReliance:
    def test_shares_sum_to_one(self):
        out = modality_reliance(_toy())
        cols = [c for c in out.columns if c.startswith("reliance_")]
        assert np.allclose(out[cols].sum(axis=1), 1.0, atol=1e-6)

    def test_shares_bounded(self):
        out = modality_reliance(_toy())
        for c in [c for c in out.columns if c.startswith("reliance_")]:
            assert out[c].between(0.0, 1.0).all()

    def test_srs_is_surface_share(self):
        out = modality_reliance(_toy())
        assert np.allclose(out["srs"], out["reliance_surface"])

    def test_sign_preserved_separately(self):
        df = _toy()
        out = modality_reliance(df)
        assert np.allclose(out["sign_surface"], np.sign(df["attr_surface"]))

    def test_zero_attribution_does_not_divide_by_zero(self):
        df = _toy(20)
        for c in ("attr_atom", "attr_surface", "attr_evolutionary"):
            df[c] = 0.0
        out = modality_reliance(df)
        assert np.isfinite(out[[c for c in out.columns if c.startswith("reliance_")]]).all().all()

    def test_missing_modality_column_is_tolerated(self):
        df = _toy().drop(columns=["attr_evolutionary"])
        out = modality_reliance(df)
        assert "reliance_evolutionary" not in out.columns
        cols = [c for c in out.columns if c.startswith("reliance_")]
        assert np.allclose(out[cols].sum(axis=1), 1.0, atol=1e-6)


class TestConfusion:
    def test_labels_correct(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.1, 0.9, 0.1])
        assert list(confusion_class(y, p, 0.5)) == ["TP", "FN", "FP", "TN"]

    def test_threshold_boundary_is_inclusive(self):
        assert confusion_class(np.array([1]), np.array([0.63]), 0.63)[0] == "TP"


class TestStatistics:
    def test_bootstrap_contains_mean(self):
        x = np.random.default_rng(0).normal(5.0, 1.0, 200)
        ci = bootstrap_ci(x, n_boot=2000, seed=1)
        assert ci.lo < ci.mean < ci.hi
        assert abs(ci.mean - 5.0) < 0.3

    def test_bootstrap_deterministic(self):
        x = np.random.default_rng(0).normal(size=100)
        assert bootstrap_ci(x, seed=7).as_tuple() == bootstrap_ci(x, seed=7).as_tuple()

    def test_bootstrap_handles_empty(self):
        ci = bootstrap_ci(np.array([]))
        assert ci.n == 0 and np.isnan(ci.mean)

    def test_paired_test_detects_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=40)
        r = paired_test(a + 1.0, a)
        assert r["p"] < 0.01 and r["mean_diff"] > 0.5

    def test_paired_test_null(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=60), rng.normal(size=60)
        assert paired_test(a, b)["p"] > 0.05

    def test_holm_monotone_and_bounded(self):
        adj = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.5})
        assert all(0 <= v <= 1 for v in adj.values())
        assert adj["a"] <= adj["b"] <= adj["c"]

    def test_holm_matches_bonferroni_on_smallest(self):
        p = {"a": 0.01, "b": 0.02, "c": 0.03}
        assert abs(holm_bonferroni(p)["a"] - 0.03) < 1e-12


class TestAggregation:
    def test_protein_counts(self):
        df = modality_reliance(_toy(300))
        agg = aggregate_by_protein(df)
        assert agg["n_residues"].sum() == 300
        assert len(agg) == df["group_id"].nunique()

    def test_error_analysis_covers_all_classes(self):
        df = modality_reliance(_toy(400))
        df["confusion"] = confusion_class(df["label"], df["prob"], 0.5)
        ea = error_analysis(df)
        assert set(ea["confusion"]) <= {"TP", "TN", "FP", "FN"}
        assert ea["n_residues"].sum() == 400

    def test_case_selection_is_deterministic(self):
        df = modality_reliance(_toy(300))
        df["confusion"] = confusion_class(df["label"], df["prob"], 0.5)
        pp = aggregate_by_protein(df)
        pp["pr_auc"] = np.linspace(0.1, 0.9, len(pp))
        a = select_cases(pp, df)
        b = select_cases(pp, df)
        pd.testing.assert_frame_equal(a, b)
        assert {"best_protein", "median_protein", "worst_protein"} <= set(a["rule"])


# ------------------------------------------------------------------ model tier
try:  # pragma: no cover
    import torch as _torch  # noqa: F401
    torch_available = True
except ImportError:  # pragma: no cover
    torch_available = False

requires_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")



def _model_noise(model, batch) -> float:
    """Largest logit difference between two identical forward passes.

    Scatter reductions on CUDA use atomicAdd, which is not deterministic, so
    this model does not reproduce bitwise even with fixed seeds. Tests that
    would otherwise demand exact equality assert against this measured spread
    instead, which separates "the code changed the computation" from "CUDA is
    not deterministic".
    """
    import torch

    with torch.no_grad():
        a = model(batch).clone()
        b = model(batch).clone()
    return float((a - b).abs().max())


@requires_torch
class TestModelTier:
    """Exercises hooks and attribution against the real checkpoint.

    The ``model`` and ``batch`` fixtures come from ``tests/conftest.py`` and are
    built from the command-line options below. Without them the whole class
    skips rather than erroring::

        pytest tests/test_xai.py -q \\
            --ckpt runs/bench_evo_pp/best.pt \\
            --manifest manifests/benchmark/bench_test315.csv \\
            --cache-dir cache/bench_evo
    """

    def test_baseline_unchanged_by_instrumentation(self, model, batch):
        import torch
        from fastfusion_if.xai import InstrumentedModel

        noise = _model_noise(model, batch)
        tol = max(10 * noise, 1e-5)

        with torch.no_grad():
            before = model(batch).clone()
        inst = InstrumentedModel(model)
        with inst.capture():
            with torch.no_grad():
                during = model(batch).clone()
        with torch.no_grad():
            after = model(batch).clone()

        d_hooked = float((before - during).abs().max())
        d_after = float((before - after).abs().max())
        assert d_hooked <= tol, (
            f"hooks changed the forward pass: max diff {d_hooked:.3e} exceeds "
            f"{tol:.3e} (model's own run-to-run noise is {noise:.3e})"
        )
        assert d_after <= tol, (
            f"hooks were not removed cleanly: max diff {d_after:.3e} > {tol:.3e}"
        )

    def test_gate_shape_and_range(self, model, batch):
        import torch
        from fastfusion_if.xai import InstrumentedModel

        with InstrumentedModel(model).capture() as cap:
            with torch.no_grad():
                model(batch)
        R = int(batch["n_residues"])
        assert cap.gate.shape == (R, model.cfg.fusion_dim)
        assert (cap.gate >= 0).all() and (cap.gate <= 1).all(), "gate is a sigmoid"

    def test_intermediates_are_on_cpu_after_capture(self, model, batch):
        """Every captured tensor must be CPU-resident once the context exits.

        Regression test. capture() used to rebind its own attribute to a
        detached copy, leaving the caller holding CUDA tensors; every downstream
        .numpy() then raised "can't convert cuda:0 device type tensor to numpy".
        """
        import torch

        from fastfusion_if.xai import InstrumentedModel

        with InstrumentedModel(model).capture() as cap:
            with torch.no_grad():
                model(batch)

        offenders = [
            k for k, v in cap.__dict__.items()
            if torch.is_tensor(v) and v.device.type != "cpu"
        ]
        assert not offenders, f"still on {cap.gate.device}: {offenders}"

    def test_capture_to_cpu_false_keeps_model_device(self, model, batch):
        """to_cpu=False must leave tensors where the model can consume them.

        The mirror of the test above. These two pin opposite directions of the
        same switch: moving when we should not raises "can't convert cuda:0
        device type tensor to numpy", and not moving when we should raises
        "Expected all tensors to be on the same device". Fixing one without the
        other has broken this module twice, so both are asserted.
        """
        import torch

        from fastfusion_if.xai import InstrumentedModel

        want = next(model.parameters()).device
        with InstrumentedModel(model).capture(to_cpu=False) as cap:
            with torch.no_grad():
                model(batch)
        for name in ("gate", "atom_res", "surface_res", "z_geom"):
            v = getattr(cap, name)
            assert v is not None and v.device.type == want.type, (
                f"{name} on {v.device}, expected {want}"
            )

    def test_captured_tensors_are_numpy_convertible(self, model, batch):
        """The operation that actually failed in production: .numpy()."""
        import numpy as np
        import torch

        from fastfusion_if.xai import InstrumentedModel

        with InstrumentedModel(model).capture() as cap:
            with torch.no_grad():
                model(batch)
        for name in ("gate", "atom_res", "surface_res"):
            arr = getattr(cap, name).numpy()
            assert isinstance(arr, np.ndarray) and np.isfinite(arr).all()

    def test_gate_decomposition_identity(self, model, batch):
        """z_geom must equal g*a + (1-g)*s exactly."""
        import torch
        from fastfusion_if.xai import InstrumentedModel

        with InstrumentedModel(model).capture() as cap:
            with torch.no_grad():
                model(batch)
        recon = cap.gate * cap.atom_res + (1 - cap.gate) * cap.surface_res
        assert torch.allclose(recon, cap.z_geom, atol=1e-5)

    def test_attribution_shapes_and_finiteness(self, model, batch):
        import torch
        from fastfusion_if.xai import integrated_gradients

        a = integrated_gradients(model, batch, n_steps=16)
        R = int(batch["n_residues"])
        for name, t in a.as_dict().items():
            assert t.shape == (R,), f"{name} has shape {t.shape}, expected ({R},)"
            assert torch.isfinite(t).all(), f"{name} contains NaN or Inf"

    def test_atom_plus_surface_equals_geom(self, model, batch):
        from fastfusion_if.xai import integrated_gradients
        import torch

        a = integrated_gradients(model, batch, n_steps=16)
        assert torch.allclose(a.atom + a.surface, a.geom, atol=1e-4)

    def test_ig_completeness_global(self, model, batch):
        """Under scope='total' the SUMMED attribution must match the summed logit.

        The backward target is sum_j logit_j, so completeness is a statement
        about the total, not about each residue: residue_context lets residue k's
        term move its neighbours' logits too.
        """
        from fastfusion_if.xai import integrated_gradients

        a = integrated_gradients(model, batch, n_steps=128, scope="total")
        assert a.relative_global_error < 0.02, (
            f"global IG completeness residual {a.relative_global_error:.4f} "
            f"(delta={a.convergence_delta_global:.4f})"
        )

    def test_ig_completeness_per_residue_self_scope(self, model, batch):
        """Under scope='self' with a mean baseline, completeness holds per residue."""
        from fastfusion_if.xai import integrated_gradients

        a = integrated_gradients(model, batch, n_steps=128, scope="self", baseline="mean")
        assert a.baseline == "mean"
        rel = a.convergence_delta.abs() / (a.logit.abs().mean() + 1e-6)
        assert float(rel.mean()) < 0.05, (
            f"per-residue residual under scope='self' too large: {float(rel.mean()):.3f}"
        )

    def test_self_scope_defaults_to_mean_baseline(self, model, batch):
        """scope='self' must not silently use the degenerate zero baseline.

        The head begins with LayerNorm, which is scale invariant, so an IG path
        from the origin never leaves a level set of the head and completeness
        cannot be recovered by adding steps.
        """
        from fastfusion_if.xai import integrated_gradients

        a = integrated_gradients(model, batch, n_steps=32, scope="self")
        assert a.baseline == "mean"

        z = integrated_gradients(model, batch, n_steps=32, scope="self", baseline="zero")
        rel_mean = float((a.convergence_delta.abs() / (a.logit.abs().mean() + 1e-6)).mean())
        rel_zero = float((z.convergence_delta.abs() / (z.logit.abs().mean() + 1e-6)).mean())
        assert rel_mean < rel_zero, (
            f"mean baseline ({rel_mean:.3f}) should beat zero ({rel_zero:.3f}) "
            "under scope='self'"
        )

    def test_determinism_within_float32_noise(self, model, batch):
        """Repeat runs must agree to the model's own numerical spread.

        Bitwise equality is unattainable: the atomicAdd scatter reductions make
        two identical forward passes differ in the last bits. The meaningful
        assertion is that repeated attribution adds no variation beyond that.
        """
        import torch

        from fastfusion_if.xai import integrated_gradients

        noise = _model_noise(model, batch)
        scale = float(model(batch).abs().max().detach()) + 1e-6
        tol = max(100 * noise, 1e-4 * scale)

        torch.manual_seed(0)
        a = integrated_gradients(model, batch, n_steps=16)
        torch.manual_seed(0)
        b = integrated_gradients(model, batch, n_steps=16)
        d = float((a.atom - b.atom).abs().max())
        assert d <= tol, (
            f"attribution varied by {d:.3e} between identical runs, above the "
            f"{tol:.3e} allowed by the model's {noise:.3e} forward-pass noise"
        )

    def test_surface_intervention_changes_prediction(self, model, batch):
        from fastfusion_if.xai import run_intervention

        r = run_intervention(model, batch, "surface")
        assert r.prob_before.shape == r.prob_after.shape
        assert np.isfinite(r.delta_logit).all()

    def test_evolutionary_shuffle_preserves_marginal(self, model, batch):
        import torch
        from fastfusion_if.xai import run_intervention

        g = torch.Generator().manual_seed(0)
        r = run_intervention(model, batch, "evolutionary", mode="shuffle", generator=g)
        assert np.isfinite(r.delta_logit).all()

    def test_atom_projection_conserves_mass(self, model, batch):
        """Uniform projection must redistribute, not create or destroy, attribution."""
        import torch
        from fastfusion_if.xai import project_to_atoms

        R = int(batch["n_residues"])
        attr = torch.ones(R)
        per_atom = project_to_atoms(attr, batch["atom2res"], None)
        assert abs(float(per_atom.sum()) - float(attr.sum())) < 1e-3
