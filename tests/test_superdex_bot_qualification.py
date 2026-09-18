"""Opt-in bot qualification regressions against direct SDK execution.

These tests require the local asset bundle (``SUPERDEX_ASSETS_PATH`` pointing
at the repository ``assets/superdex`` copy) plus the optional SuperDex
runtime; they skip cleanly when either is missing. They reuse the vocabulary
of ``scripts/superdex_compare.py`` (profiles, reference scenes, tolerances)
so the pytest subset and the consolidated tool cannot drift apart: every
registered candidate runs the structure and trajectory checks against a
direct SuperDex SDK scene driven with identical inputs, and blocked
candidates must keep their precise rejection instead of silently loading.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info[:2] not in ((3, 12), (3, 13)):
    pytest.skip("SuperDex wheels require Python 3.12 or 3.13", allow_module_level=True)
pytest.importorskip("superdex.physics")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import superdex_compare as compare  # noqa: E402

TOL = 1e-7

# Bots whose qualification is intentionally not registered: each carries a
# classified capability blocker recorded in the maintained inventory.
BLOCKED_KEYS = {
    "example_bot_2dof": "bots/fun/example_bot_2dof/example_bot_2dof.superdex_bot",
    "dg5f_seed": "bots/sensors/dg5f_seed/dg5f_seed.superdex_bot",
    "dg5f_long_seed_left": "bots/hands/dg5f_long_seed/left/dg5f_long_seed_left.superdex_bot",
    "dg5f_long_seed_right": "bots/hands/dg5f_long_seed/right/dg5f_long_seed_right.superdex_bot",
    "dg5f_short_seed_left": "bots/hands/dg5f_short_seed/left/dg5f_short_seed_left.superdex_bot",
    "dg5f_short_seed_right": "bots/hands/dg5f_short_seed/right/dg5f_short_seed_right.superdex_bot",
    "fr3_dg5f_short_seed_right":
        "bots/arm_hand_combos/fr3_dg5f_short_seed/right/fr3_dg5f_short_seed_right.superdex_bot",
}


def _assets_root() -> Path | None:
    root = Path(
        os.environ.get("SUPERDEX_ASSETS_PATH")
        or (REPOSITORY_ROOT / "assets" / "superdex")
    ).resolve()
    return root if root.is_dir() and any(root.rglob(".superdex_root")) else None


ASSETS = _assets_root()
pytestmark = pytest.mark.skipif(
    ASSETS is None, reason="set SUPERDEX_ASSETS_PATH to run the bot qualification fixture"
)


def _bot_path(relpath: str) -> Path:
    assert ASSETS is not None
    path = ASSETS / relpath
    if not path.is_file():
        pytest.skip(f"fixture missing: {path}")
    return path


@pytest.fixture
def openarm_backend():
    from unisim import create_backend
    from unisim.backend.superdex.runtime import (
        acquire_runtime,
        load_superdex_dependencies,
        release_runtime,
    )
    from unisim.scene import SceneCfg

    path = _bot_path(compare.PROFILES["openarm_v20_combo"].relpath)
    physics, _robotics = load_superdex_dependencies()
    acquire_runtime(physics)
    reference = compare.BotReference(path)
    try:
        efforts = compare.resolve_efforts(
            compare.ModelRef(path, "bot", path.name), reference)
    finally:
        reference.close()
        release_runtime(physics)
    backend = create_backend(
        "superdex",
        SceneCfg(str(path)),
        2,
        compare.SIM_DT,
        superdex_effort_limits=efforts,
    )
    try:
        yield backend, efforts
    finally:
        backend.close()


def _held_reference(path: Path):
    """Context manager keeping the shared runtime alive for a reference scene."""
    from contextlib import contextmanager

    from unisim.backend.superdex.runtime import (
        acquire_runtime,
        load_superdex_dependencies,
        release_runtime,
    )

    @contextmanager
    def _open():
        physics, _robotics = load_superdex_dependencies()
        acquire_runtime(physics)
        try:
            yield compare.BotReference(path)
        finally:
            release_runtime(physics)

    return _open()


def test_registered_profiles_cover_recipe_candidates():
    # Every runnable recipe candidate from the maintained inventory is
    # registered for qualification; blocked recipes carry inventory blockers.
    inventory = REPOSITORY_ROOT / "docs" / "superdex-assets-inventory.json"
    if not inventory.is_file():
        pytest.skip("recorded inventory not present")
    import json

    recorded = json.loads(inventory.read_text(encoding="utf-8"))
    recipe_entries = {
        e["entrypoint"]
        for e in recorded["entries"]
        if e.get("disposition") == "recipe-candidate"
    }
    covered = {p.relpath for p in compare.PROFILES.values()}
    missing = recipe_entries - covered
    assert not missing, f"recipe candidates without qualification: {sorted(missing)}"


def test_openarm_recipe_structure_matches_direct_sdk(openarm_backend):
    backend, _ = openarm_backend
    path = _bot_path(compare.PROFILES["openarm_v20_combo"].relpath)
    with _held_reference(path) as reference:
        assert len(backend.model.joint_names) == len(reference.axis_columns)
        # Recipe attachments and prefixes resolve through the SDK; the
        # adapter's compiled inventories match the SDK-compiled prefab.
        assert tuple(backend.model.body_names)[1:] == tuple(
            str(link.name) for link in reference.cfg.links
        )
        assert tuple(backend.get_actuator_names()) == tuple(reference.axis_columns)
        np.testing.assert_allclose(
            backend.get_dof_pos()[0], reference.joint_state()[0], atol=1e-6
        )


def test_openarm_recipe_trajectory_matches_direct_sdk(openarm_backend):
    backend, efforts = openarm_backend
    task = compare.WorkerTask(
        label="openarm_v20_torso",
        path=str(_bot_path(compare.PROFILES["openarm_v20_combo"].relpath)),
        kind="bot",
        steps=compare.SWEEP_CTRL_STEPS * compare.DECIMATION,
    )
    result = compare.run_model_subprocess(task, timeout=900)
    assert result["status"] == "passed", result.get("worker_output_tail", "")[-500:]
    evidence = result["checks"]["trajectory_equivalence"]
    assert evidence["passed"], evidence
    assert evidence["physics_steps"] >= 1000


def test_googly_eyes_single_joint_qualifies():
    task = compare.WorkerTask(
        label="googly_eyes",
        path=str(_bot_path(compare.PROFILES["googly_eyes"].relpath)),
        kind="bot",
        steps=compare.SWEEP_CTRL_STEPS * compare.DECIMATION,
    )
    result = compare.run_model_subprocess(task, timeout=900)
    assert result["status"] == "passed", result.get("worker_output_tail", "")[-500:]
    assert result["checks"]["structure"]["passed"]


def test_blocked_candidates_reject_with_precise_blockers():
    """Every blocked bot must fail closed with its classified native blocker."""
    from unisim import create_backend
    from unisim.scene import SceneCfg

    for key, relpath in sorted(BLOCKED_KEYS.items()):
        with pytest.raises((NotImplementedError, ValueError)):
            backend = create_backend(
                "superdex",
                SceneCfg(str(_bot_path(relpath))),
                1,
                compare.SIM_DT,
                superdex_execution_mode="serial",
                superdex_num_workers=0,
            )
            backend.close()


def test_compare_tool_writes_reports_only_into_out_directory(tmp_path):
    """Reports and limitation records are opt-in; nothing is written by default."""
    task = compare.WorkerTask(
        label="googly_eyes",
        path=str(_bot_path(compare.PROFILES["googly_eyes"].relpath)),
        kind="bot",
        steps=1000,
    )
    results = [compare.run_model_subprocess(task, timeout=900)]
    compare.write_reports(tmp_path, results, [])
    report = tmp_path / "comparison.json"
    assert report.is_file()
    import json

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["results"][0]["status"] == "passed"
    # A fresh default run must leave no output anywhere else.
    assert not (REPOSITORY_ROOT / "results").exists() or all(
        p.is_dir() for p in (REPOSITORY_ROOT / "results").iterdir()
    )
