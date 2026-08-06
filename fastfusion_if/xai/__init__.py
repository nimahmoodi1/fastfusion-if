"""Optional explainability layer for FastFusion-IF.

Importing this package must never be required for training or inference. The
torch-dependent modules are imported lazily so that the statistical layer
(:mod:`analysis`) is usable without torch installed.
"""
from .analysis import (  # noqa: F401
    BootstrapCI,
    aggregate_by_protein,
    bootstrap_ci,
    confusion_class,
    error_analysis,
    holm_bonferroni,
    modality_reliance,
    paired_test,
    select_cases,
)

__all__ = [
    "modality_reliance", "confusion_class", "aggregate_by_protein",
    "bootstrap_ci", "BootstrapCI", "paired_test", "holm_bonferroni",
    "select_cases", "error_analysis",
]


def __getattr__(name):  # lazy torch-dependent exports
    if name in {"InstrumentedModel", "Intermediates", "modality_availability"}:
        from . import hooks
        return getattr(hooks, name)
    if name in {"integrated_gradients", "ResidueAttribution", "project_to_atoms", "project_to_surface"}:
        from . import attribution
        return getattr(attribution, name)
    if name in {"run_intervention", "deletion_curve", "faithfulness_gap", "InterventionResult"}:
        from . import interventions
        return getattr(interventions, name)
    raise AttributeError(name)
