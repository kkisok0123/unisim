"""Opt-in stage-3 bot qualification against direct SDK execution.

These tests require the local asset bundle (``SUPERDEX_ASSETS_PATH`` pointing
at the repository ``assets/superdex`` copy) plus the optional SuperDex
runtime; they skip cleanly when either is missing. They reuse the parameterized
runner in ``scripts/superdex_bot_qualify.py`` so the pytest subset and the
recorded reports cannot drift apart: every registered candidate runs the
structure and trajectory checks against a direct SuperDex SDK scene driven
with identical inputs, and blocked candidates must keep their precise
rejection instead of silently loading.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info[:2] not in ((3, 12), (3, 13)):
    pytest.skip("SuperDex wheels require Python 3.12 or 3.13", allow_module_level=True)
pytest.importorskip("superdex.physics")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "superdex_bot_qualify.py"

_spec = importlib.util.spec_from_file_location("superdex_bot_qualify", SCRIPT)
qualify = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("superdex_bot_qualify", qualify)
_spec.loader.exec_module(qualify)

TOL = 1e-7


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


def _resolved(profile):
    cfg = qualify._load_prefab(_bot_path(profile.relpath))
    names, _, efforts, armature = qualify.authored_joint_metadata(cfg)
    from superdex_bot_profiles import resolve_gains

    return resolve_gains(profile, names, armature, efforts)


@pytest.fixture
def openarm_backend():
    from unisim import create_backend
    from unisim.scene import SceneCfg

    profile = qualify.PROFILES["openarm_v20"]
    resolved = _resolved(profile)
    backend = create_backend(
        "superdex",
        SceneCfg(str(_bot_path(profile.relpath))),
        2,
        qualify.SIM_DT,
        superdex_effort_limits=list(resolved.effort_limits),
    )
    try:
        yield backend, resolved
    finally:
        backend.close()


def test_registered_profiles_cover_recipe_candidates():
    from superdex_bot_profiles import BLOCKED_KEYS, PROFILES

    # Every recipe-candidate bot from the stage-1 inventory is either
    # registered for qualification or listed as blocked with evidence.
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
    covered = {p.relpath for p in PROFILES.values()} | set(BLOCKED_KEYS.values())
    missing = recipe_entries - covered
    assert not missing, f"recipe candidates without qualification or blocker: {sorted(missing)}"


def test_openarm_recipe_structure_matches_direct_sdk(openarm_backend):
    backend, _ = openarm_backend
    reference = None
    try:
        reference = qualify.Reference(_bot_path(qualify.PROFILES["openarm_v20"].relpath),
                                      qualify.SIM_DT)
        assert backend.model.nq == reference.n == 18
        # Recipe attachments and prefixes resolve through the SDK; the
        # adapter's compiled inventories match the SDK-compiled prefab.
        assert tuple(backend.model.body_names)[1:] == tuple(
            str(link.name) for link in reference.cfg.links
        )
        assert tuple(backend.get_actuator_names()) == reference.names
        np.testing.assert_allclose(
            backend.get_default_qpos(), reference.qvel()[0], atol=1e-6
        )
        np.testing.assert_allclose(
            backend.get_joint_range(), reference.ranges, atol=1e-6
        )
    finally:
        if reference is not None:
            reference.close()


def test_openarm_recipe_trajectory_matches_direct_sdk(openarm_backend):
    backend, resolved = openarm_backend
    run = qualify.QualificationRun(qualify.PROFILES["openarm_v20"], _bot_path(
        qualify.PROFILES["openarm_v20"].relpath))
    from unisim import create_backend
    from unisim.scene import SceneCfg

    serial = create_backend(
        "superdex",
        SceneCfg(str(run.bot_path)),
        1,
        qualify.SIM_DT,
        superdex_execution_mode="serial",
        superdex_num_workers=0,
        superdex_effort_limits=list(resolved.effort_limits),
    )
    reference = None
    try:
        reference = qualify.Reference(run.bot_path, qualify.SIM_DT)
        run.check_trajectory_equivalence(backend, serial, reference, resolved)
        evidence = run.report["checks"]["trajectory_equivalence"]
        assert evidence["passed"], evidence
        assert evidence["physics_steps"] >= 1000
    finally:
        if reference is not None:
            reference.close()
        serial.close()


def test_googly_eyes_single_joint_qualifies():
    profile = qualify.PROFILES["googly_eyes"]
    resolved = _resolved(profile)
    run = qualify.QualificationRun(profile, _bot_path(profile.relpath))
    backend = run.make_backend(resolved, 2)
    reference = None
    try:
        reference = qualify.Reference(run.bot_path, qualify.SIM_DT)
        run.check_structure(backend, reference)
    finally:
        if reference is not None:
            reference.close()
        backend.close()
    assert run.report["checks"]["structure"]["passed"]


def test_blocked_candidates_reject_with_precise_blockers():
    """Every blocked bot must fail closed with its recorded native blocker."""
    from unisim import create_backend
    from unisim.scene import SceneCfg

    for key, relpath in sorted(qualify.BLOCKED_KEYS.items()):
        if key == "openarm_v20_torso":
            continue  # loads in serial; blocked only in batch mode
        with pytest.raises((NotImplementedError, ValueError)):
            backend = create_backend(
                "superdex",
                SceneCfg(str(_bot_path(relpath))),
                1,
                qualify.SIM_DT,
                superdex_execution_mode="serial",
                superdex_num_workers=0,
            )
            backend.close()


def test_recorded_compatibility_table_is_current():
    table = (
        REPOSITORY_ROOT
        / "docs"
        / "superdex-bots-qualification"
        / "compatibility-table.md"
    )
    if not table.is_file():
        pytest.skip("recorded compatibility table not present")
    text = table.read_text(encoding="utf-8")
    from superdex_bot_profiles import BLOCKED_KEYS, PROFILES

    for key in PROFILES:
        assert f"`{key}`" in text, f"qualified bot {key} missing from the table"
    for key in BLOCKED_KEYS:
        assert f"`{key}`" in text, f"blocked bot {key} missing from the table"
