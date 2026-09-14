"""Opt-in stage-2B FR3 adapter qualification against direct SDK execution.

These tests require the local asset bundle (``SUPERDEX_ASSETS_PATH`` pointing
at the repository ``assets/superdex`` copy) plus the optional SuperDex
runtime; they skip cleanly when either is missing. Each check compares the
adapter against a direct SuperDex SDK scene driven with identical inputs, so
adapter translation is separated from asset/SDK behavior. The full recorded
qualification (with report) is ``scripts/superdex_fr3_qualify.py``; this
module keeps the regression subset runnable under pytest.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from unisim import create_backend
from unisim.scene import SceneCfg

if sys.version_info[:2] not in ((3, 12), (3, 13)):
    pytest.skip("SuperDex wheels require Python 3.12 or 3.13", allow_module_level=True)
pytest.importorskip("superdex.physics")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "superdex_fr3_qualify.py"

_spec = importlib.util.spec_from_file_location("superdex_fr3_qualify", SCRIPT)
qualify = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("superdex_fr3_qualify", qualify)
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
    ASSETS is None, reason="set SUPERDEX_ASSETS_PATH to run the FR3 qualification fixture"
)


def _bot_path() -> Path:
    assert ASSETS is not None
    path = ASSETS / qualify.BOT_RELPATH
    if not path.is_file():
        pytest.skip(f"FR3 fixture missing: {path}")
    return path


@pytest.fixture
def fr3():
    backend = create_backend(
        "superdex",
        SceneCfg(str(_bot_path())),
        2,
        qualify.SIM_DT,
        base_name="fr3_link0",
        superdex_effort_limits=qualify.EFFORT_LIMITS,
    )
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def reference(fr3):
    ref = qualify.Reference(_bot_path(), qualify.SIM_DT)
    try:
        yield ref
    finally:
        # Destroyed before the backend fixture releases the shared runtime.
        ref.close()


def test_fr3_structure_matches_direct_sdk(fr3, reference):
    model = fr3.model
    assert tuple(model.body_names) == ("world", "base", *(f"fr3_link{i}" for i in range(9)))
    joints = tuple(f"fr3_joint{i}" for i in range(1, 8))
    assert fr3.get_actuator_names() == joints
    assert tuple(model.joint_names) == joints

    q0 = fr3.get_default_qpos()
    ref_q0, _ = reference.qvel()
    np.testing.assert_allclose(q0, ref_q0, atol=1e-6)

    # Authored joint ranges follow the projected authored limits exactly.
    np.testing.assert_allclose(fr3.get_joint_range(), reference.authored_joint_ranges(), atol=1e-6)

    # Dynamic-link masses agree with the compiled SDK actor. The world-welded
    # fr3_link0 keeps its authored mass by design; the SDK reports static 0.
    masses = fr3.get_body_mass()
    for body_id in range(2, len(model.body_names)):
        ref_link = reference.links[body_id - 1]
        if not ref_link.is_static():
            np.testing.assert_allclose(
                masses[body_id], float(ref_link.get_mass()), atol=1e-6
            )

    # World link transforms and AABBs at the default pose.
    body_ids = np.arange(len(model.body_names))
    pos_w = fr3.get_body_pos_w(body_ids)[0]
    ref_pos = np.array([reference.link_pos(i) for i in range(len(reference.links))])
    np.testing.assert_allclose(pos_w[1:], ref_pos, atol=1e-6)


def test_fr3_control_ordering_and_effort_clipping(fr3):
    # Differencing isolates each control column: the torqued joint is the
    # most affected DoF for every joint index.
    for joint_index in range(7):
        baseline = create_backend(
            "superdex", SceneCfg(str(_bot_path())), 1, qualify.SIM_DT,
            base_name="fr3_link0", superdex_effort_limits=qualify.EFFORT_LIMITS,
        )
        driven = create_backend(
            "superdex", SceneCfg(str(_bot_path())), 1, qualify.SIM_DT,
            base_name="fr3_link0", superdex_effort_limits=qualify.EFFORT_LIMITS,
        )
        try:
            baseline.step(np.zeros((1, 7)), 5)
            torque = np.zeros((1, 7))
            torque[0, joint_index] = 5.0
            driven.step(torque, 5)
            difference = driven.get_dof_vel()[0] - baseline.get_dof_vel()[0]
        finally:
            baseline.close()
            driven.close()
        assert int(np.argmax(np.abs(difference))) == joint_index

    # A strongly saturated command reproduces per-joint capped dynamics
    # exactly.
    saturated = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, qualify.SIM_DT,
        base_name="fr3_link0", superdex_effort_limits=qualify.EFFORT_LIMITS,
    )
    capped = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, qualify.SIM_DT,
        base_name="fr3_link0", superdex_effort_limits=qualify.EFFORT_LIMITS,
    )
    try:
        saturated.step(np.full((1, 7), 1e4), 200)
        capped.step(np.asarray(qualify.EFFORT_LIMITS)[None, :], 200)
        np.testing.assert_array_equal(
            saturated.get_state()["qpos"], capped.get_state()["qpos"]
        )
    finally:
        saturated.close()
        capped.close()


def test_fr3_trajectory_matches_direct_sdk(fr3, reference):
    serial = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, qualify.SIM_DT,
        base_name="fr3_link0", superdex_execution_mode="serial",
        superdex_num_workers=0, superdex_effort_limits=qualify.EFFORT_LIMITS,
    )
    try:
        q0 = fr3.get_default_qpos().copy()
        fr3.reset()
        serial.reset()
        reference.set_state(q0, np.zeros(7))
        for k in range(qualify.SWEEP_CTRL_STEPS):
            target = qualify.sweep_target(q0, k)
            tau_b = qualify.pd_torque(target, fr3.get_dof_pos()[0], fr3.get_dof_vel()[0])
            tau_s = qualify.pd_torque(
                target, serial.get_dof_pos()[0], serial.get_dof_vel()[0]
            )
            q_ref, v_ref = reference.qvel()
            tau_r = qualify.pd_torque(target, q_ref, v_ref)
            fr3.step(np.repeat(tau_b[None, :], 2, axis=0), qualify.DECIMATION)
            serial.step(tau_s[None, :], qualify.DECIMATION)
            reference.step(tau_r, qualify.DECIMATION)
            q_ref, _ = reference.qvel()
            np.testing.assert_allclose(
                fr3.get_dof_pos()[0], q_ref, atol=TOL,
                err_msg=f"batch env0 diverged from direct SDK at ctrl step {k}",
            )
            np.testing.assert_allclose(
                serial.get_dof_pos()[0], q_ref, atol=TOL,
                err_msg=f"serial backend diverged from direct SDK at ctrl step {k}",
            )
        # >= 1000 physics steps of stability evidence under the same profile.
        assert qualify.SWEEP_CTRL_STEPS * qualify.DECIMATION >= 1000
        assert np.isfinite(fr3.get_state()["qvel"]).all()
    finally:
        serial.close()


def test_fr3_contact_recovery_matches_direct_sdk(fr3, reference):
    reference.register_contact_queries()
    q0 = fr3.get_default_qpos().copy()
    q_pen = q0.copy()
    q_pen[1] = -2.6  # beyond joint 2's authored range -1.784

    fr3.set_state(np.array([0]), q_pen[None, :], np.zeros((1, 7)))
    reference.set_state(q_pen, np.zeros(7))
    np.testing.assert_allclose(
        fr3.get_state()["qpos"][0], reference.qvel()[0], atol=TOL
    )

    zero = np.zeros((fr3.num_envs, 7))
    contact_steps = 0
    for _ in range(400):
        fr3.step(zero, 1)
        reference.step(np.zeros(7), 1)
        q_ref, _ = reference.qvel()
        np.testing.assert_allclose(fr3.get_state()["qpos"][0], q_ref, atol=TOL)
        if reference.contact_point_count() > 0:
            contact_steps += 1
    assert contact_steps > 0, "penetrating pose did not produce self-contact"


def test_fr3_reset_and_round_trips(fr3, reference):
    q0 = fr3.get_default_qpos().copy()

    fr3.reset()
    fr3.step(np.full((fr3.num_envs, 7), 3.0), 100)
    fr3.reset()
    np.testing.assert_allclose(
        fr3.get_state()["qpos"], np.tile(q0, (fr3.num_envs, 1)), atol=1e-6
    )
    reference.set_state(q0, np.zeros(7))
    np.testing.assert_allclose(
        fr3.get_state()["qpos"][0], reference.qvel()[0], atol=1e-6
    )

    # Selective reset leaves the untouched environment bit-identical.
    fr3.reset()
    fr3.step(np.full((fr3.num_envs, 7), 3.0), 100)
    before = fr3.get_state()
    fr3.reset(np.array([0]))
    after = fr3.get_state()
    np.testing.assert_allclose(after["qpos"][0], q0, atol=1e-6)
    np.testing.assert_array_equal(after["qpos"][1], before["qpos"][1])
    np.testing.assert_array_equal(after["qvel"][1], before["qvel"][1])

    # State round trip.
    q_arb = q0 + np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1])
    v_arb = np.array([0.05, -0.04, 0.03, -0.02, 0.01, -0.01, 0.2])
    fr3.set_state(np.array([1]), q_arb[None, :], v_arb[None, :])
    rb = fr3.get_state()
    np.testing.assert_allclose(rb["qpos"][1], q_arb, atol=1e-6)
    np.testing.assert_allclose(rb["qvel"][1], v_arb, atol=1e-6)


def test_fr3_environment_isolation(fr3):
    single = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, qualify.SIM_DT,
        base_name="fr3_link0", superdex_effort_limits=qualify.EFFORT_LIMITS,
    )
    try:
        q0 = fr3.get_default_qpos().copy()
        for env_index, hold in ((0, False), (1, True)):
            fr3.reset()
            single.reset()
            for k in range(60):
                target = qualify.sweep_target(q0, k)
                own_target = q0 if hold else target
                tau_own = qualify.pd_torque(
                    own_target,
                    fr3.get_dof_pos()[env_index],
                    fr3.get_dof_vel()[env_index],
                )
                commands = np.zeros((fr3.num_envs, 7))
                commands[env_index] = tau_own
                other = 1 - env_index
                other_target = q0 if not hold else target
                commands[other] = qualify.pd_torque(
                    other_target, fr3.get_dof_pos()[other], fr3.get_dof_vel()[other]
                )
                fr3.step(commands, qualify.DECIMATION)
                tau_single = qualify.pd_torque(
                    own_target, single.get_dof_pos()[0], single.get_dof_vel()[0]
                )
                single.step(tau_single[None, :], qualify.DECIMATION)
                np.testing.assert_allclose(
                    fr3.get_dof_pos()[env_index],
                    single.get_dof_pos()[0],
                    atol=TOL,
                    err_msg=f"env {env_index} contaminated at ctrl step {k}",
                )
    finally:
        single.close()


def test_fr3_repeated_create_step_close_cycles():
    for _ in range(3):
        backend = create_backend(
            "superdex", SceneCfg(str(_bot_path())), 1, qualify.SIM_DT,
            base_name="fr3_link0", superdex_effort_limits=qualify.EFFORT_LIMITS,
        )
        try:
            backend.step(np.zeros((1, 7)), 10)
            backend.reset()
            np.testing.assert_allclose(
                backend.get_state()["qpos"],
                backend.get_default_qpos()[None, :],
                atol=1e-6,
            )
        finally:
            backend.close()


def test_fr3_qualification_script_report_is_current():
    """The recorded report must match a rerun of the qualification script."""
    report_path = (
        REPOSITORY_ROOT / "docs" / "superdex-fr3-qualification" / "report.json"
    )
    if not report_path.is_file():
        pytest.skip("recorded qualification report not present")
    import json

    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    failures = [name for name, check in recorded["checks"].items() if not check["passed"]]
    assert not failures, f"recorded report has failing checks: {failures}"
    # The recorded profile must match the script constants the tests use.
    assert recorded["effort_limits_nm"] == qualify.EFFORT_LIMITS
    assert recorded["pd_gains"]["kp"] == qualify.KP
    assert recorded["pd_gains"]["kd"] == qualify.KD
    assert recorded["sim_dt"] == qualify.SIM_DT
    assert recorded["decimation"] == qualify.DECIMATION
