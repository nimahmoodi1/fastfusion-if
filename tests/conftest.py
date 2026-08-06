"""pytest configuration for the XAI tests.

Two problems this file solves, both seen on the development machine.

**Missing fixtures.** The model-tier tests need a loaded checkpoint and a real
batch. Those cannot be constructed from nothing, so they come from command-line
options. Without them the model tier *skips* rather than erroring, which is the
behaviour a test suite should have when an optional resource is absent::

    pytest tests/test_xai.py -q                       # statistical tier only
    pytest tests/test_xai.py -q \\
        --ckpt runs/bench_evo_pp/best.pt \\
        --manifest manifests/benchmark/bench_test315.csv \\
        --cache-dir cache/bench_evo                   # full suite

**ROS 2 plugin collision.** A machine with ROS 2 Humble installed has
``launch_testing_ros`` on the pytest plugin path. It registers a hook
(``pytest_launch_collect_makemodule``) that recent pytest does not know, which
crashes collection with ``PluginValidationError``. Autoloading of third-party
plugins is therefore disabled here; the plugins this suite actually needs are
requested explicitly. If you prefer to do it from the shell instead, use::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_xai.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Put the repository root on sys.path before any test module is imported.
#
# Under pytest's default "prepend" import mode, the directory inserted into
# sys.path is the test file's basedir -- `tests/`, since there is no
# `tests/__init__.py`. The repository root is NOT added. `python -m pytest`
# happens to work because Python itself prepends the CWD; the bare `pytest`
# console script does not, which is why one worked and the other raised
# ModuleNotFoundError: No module named 'fastfusion_if'.
#
# conftest.py is imported during collection, before test modules, so inserting
# here fixes both invocations. `pytest.ini` also sets `pythonpath = .` for
# pytest >= 7; this block covers older versions and direct conftest use.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must be set before pytest loads entry-point plugins. Setting it inside
# conftest works because conftest is imported during startup, before the
# entry-point scan for the rootdir's plugins in the common case; the env var in
# `pytest.ini` covers the rest.
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def pytest_addoption(parser):
    g = parser.getgroup("xai")
    g.addoption("--ckpt", action="store", default=None,
                help="checkpoint for the model-tier tests, e.g. runs/bench_evo_pp/best.pt")
    g.addoption("--manifest", action="store", default=None,
                help="manifest CSV supplying the test protein")
    g.addoption("--split", action="store", default="test")
    g.addoption("--cache-dir", action="store", default=None,
                help="precomputed feature cache, e.g. cache/bench_evo")
    g.addoption("--pdb-root", action="store", default=None,
                help="prefix for relative manifest paths; needed with the committed "
                     "manifests, whose paths were sanitised to be relative")


def _require(config, name: str) -> str:
    v = config.getoption(name)
    if not v:
        pytest.skip(
            f"model tier needs --{name.lstrip('-').replace('_', '-')}; "
            "pass --ckpt, --manifest and --cache-dir to run it"
        )
    return v


@pytest.fixture(scope="session")
def device():
    torch = pytest.importorskip("torch")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def loaded(request, device):
    """Load the checkpoint once per session; it is the expensive part."""
    pytest.importorskip("torch")
    from run_xai import iter_proteins, load_model  # sys.path set at module import

    ckpt = _require(request.config, "ckpt")
    manifest = _require(request.config, "manifest")
    cache_dir = _require(request.config, "cache_dir")
    split = request.config.getoption("split")

    pdb_root = request.config.getoption("pdb_root")

    model, cfg, ck = load_model(ckpt, device)
    batch = next(iter_proteins(manifest, split, cache_dir, cfg, pdb_root=pdb_root))
    return model, batch, cfg, ck


@pytest.fixture
def model(loaded):
    return loaded[0]


@pytest.fixture
def batch(loaded, device):
    import torch

    b = loaded[1]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
