"""FR3 adapter regression against direct SDK execution.

These tests require the local asset bundle (``SUPERDEX_ASSETS_PATH`` pointing
at the repository ``assets/superdex`` copy) plus the optional SuperDex
runtime; they skip cleanly when either is missing. Each check compares the
adapter against a direct SuperDex SDK scene driven with identical inputs
through the shared vocabulary of ``scripts/superdex_compare.py`` (profiles,
reference scenes, tolerances), so adapter translation is separated from
asset/SDK behavior. The consolidated numerical comparison lives in that
script; this module keeps the regression subset runnable under pytest.
"""

from __future__ import annotations

import math
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
SCRIPTS = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import superdex_compare as compare  # noqa: E402

TOL = 1e-7
BOT_RELPATH = "bots/arms/fr3_v2/fr3_v2.superdex_bot"
EFFORT_LIMITS = (87, 87, 87, 87, 12, 12, 12)
KP = 400.0
SWEEP_CTRL_STEPS = 130
CONTACT_PENETRATION = 0.8
CONTACT_STEPS = 400


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
    path = ASSETS / BOT_RELPATH
    if not path.is_file():
        pytest.skip(f"FR3 fixture missing: {path}")
    return path


@pytest.fixture
def fr3():
    backend = create_backend(
        "superdex",
        SceneCfg(str(_bot_path())),
        2,
        compare.SIM_DT,
        base_name="fr3_link0",
        superdex_effort_limits=list(EFFORT_LIMITS),
    )
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def reference(fr3):
    ref = compare.BotReference(_bot_path(), compare.SIM_DT)
    try:
        yield ref
    finally:
        # Destroyed before the backend fixture releases the shared runtime.
        ref.close()


def _armature(fr3) -> np.ndarray:
    return fr3.get_dof_armature()[fr3.model.joint_qvel_indices]


def _kd(fr3) -> np.ndarray:
    return 2.0 * np.sqrt(KP * np.maximum(_armature(fr3), 1e-3))


def _sweep_target(q0: np.ndarray, amplitudes: np.ndarray, k: int) -> np.ndarray:
    t = k * compare.SIM_DT * compare.DECIMATION
    phase = np.linspace(0.0, 3.0 * math.pi / 2.0, len(q0))
    return q0 + amplitudes * np.sin(2.0 * math.pi * t / 2.0 + phase)


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
    np.testing.assert_allclose(fr3.get_joint_range(), reference.ranges, atol=1e-6)

    # Dynamic-link masses agree with the compiled SDK actor. The world-welded
    # fr3_link0 keeps its authored mass by design; the SDK reports static 0.
    masses = fr3.get_body_mass()
    for body_id in range(2, len(model.body_names)):
        ref_link = reference.links[body_id - 1]
        if not ref_link.is_static():
            np.testing.assert_allclose(
                masses[body_id], float(ref_link.get_mass()), atol=1e-6
            )

    # World link transforms at the default pose.
    body_ids = np.arange(len(model.body_names))
    pos_w = fr3.get_body_pos_w(body_ids)[0]
    ref_pos = np.array([reference.link_pos(i) for i in range(len(reference.links))])
    np.testing.assert_allclose(pos_w[1:], ref_pos, atol=1e-6)


def test_fr3_control_ordering_and_effort_clipping(fr3):
    n = len(EFFORT_LIMITS)
    probe = 5.0
    fr3.reset()
    fr3.step(np.zeros((2, n)), 5)
    base = fr3.get_dof_vel()[0].copy()
    for i in range(n):
        fr3.reset()
        torque = np.zeros((2, n))
        torque[:, i] = probe
        fr3.step(torque, 5)
        response = fr3.get_dof_vel() - base[None, :]
        # Each control column reaches its own joint.
        assert response[0, i] != 0.0

    limits = np.asarray(EFFORT_LIMITS)
    saturated = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, compare.SIM_DT,
        superdex_effort_limits=list(EFFORT_LIMITS))
    capped = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, compare.SIM_DT,
        superdex_effort_limits=list(EFFORT_LIMITS))
    try:
        saturated.step(np.full((1, n), 1e4), 200)
        capped.step(limits[None, :].astype(float), 200)
        np.testing.assert_array_equal(
            saturated.get_state()["qpos"], capped.get_state()["qpos"])
    finally:
        saturated.close()
        capped.close()


def test_fr3_trajectory_matches_direct_sdk(fr3, reference):
    n = len(EFFORT_LIMITS)
    kd = _kd(fr3)
    effort = np.asarray(EFFORT_LIMITS)
    q0 = fr3.get_default_qpos().copy()
    amplitudes = np.asarray(compare.sweep_amplitudes(
        compare.PROFILES["fr3_v2"], q0, fr3.get_joint_range()))
    fr3.reset()
    reference.set_joint_state(q0, np.zeros(n))
    batch_dev = serial_dev = 0.0
    serial = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, compare.SIM_DT,
        superdex_execution_mode="serial", superdex_num_workers=0,
        superdex_effort_limits=list(EFFORT_LIMITS))
    try:
        serial.reset()
        for k in range(SWEEP_CTRL_STEPS):
            target = _sweep_target(q0, amplitudes, k)
            tau_b = compare.pd_torque(KP, kd, effort, target,
                                      fr3.get_dof_pos()[0], fr3.get_dof_vel()[0])
            tau_s = compare.pd_torque(KP, kd, effort, target,
                                      serial.get_dof_pos()[0], serial.get_dof_vel()[0])
            q_ref, v_ref = reference.joint_state()
            tau_r = compare.pd_torque(KP, kd, effort, target, q_ref, v_ref)
            fr3.step(np.repeat(tau_b[None, :], 2, axis=0), compare.DECIMATION)
            serial.step(tau_s[None, :], compare.DECIMATION)
            reference.step(tau_r, compare.DECIMATION)
            q_ref, _ = reference.joint_state()
            batch_dev = max(batch_dev, float(np.max(np.abs(fr3.get_dof_pos()[0] - q_ref))))
            serial_dev = max(serial_dev, float(np.max(np.abs(serial.get_dof_pos()[0] - q_ref))))
    finally:
        serial.close()
    assert batch_dev <= TOL and serial_dev <= TOL
    assert SWEEP_CTRL_STEPS * compare.DECIMATION >= 1000


def test_fr3_contact_recovery_matches_direct_sdk(fr3, reference):
    n = len(EFFORT_LIMITS)
    joint = 1
    lo = float(reference.ranges[joint][0])
    probes = reference.register_contact_queries()
    q0 = fr3.get_default_qpos().copy()
    q_pen = q0.copy()
    q_pen[joint] = lo - CONTACT_PENETRATION
    fr3.set_state(np.array([0]), q_pen[None, :], np.zeros((1, n)))
    reference.set_joint_state(q_pen, np.zeros(n))
    contact_steps = 0
    max_dev = 0.0
    for _ in range(CONTACT_STEPS):
        fr3.step(np.zeros((2, n)), 1)
        reference.step(np.zeros(n), 1)
        q_ref, _ = reference.joint_state()
        max_dev = max(max_dev, float(np.max(np.abs(fr3.get_state()["qpos"][0] - q_ref))))
        if reference.contact_point_count(probes) > 0:
            contact_steps += 1
    assert contact_steps > 0
    assert max_dev <= TOL


def test_fr3_reset_and_round_trips(fr3):
    n = len(EFFORT_LIMITS)
    probe = min(EFFORT_LIMITS) * 0.5
    q0 = fr3.get_default_qpos().copy()
    fr3.reset()
    fr3.step(np.full((2, n), probe), 100)
    fr3.reset()
    np.testing.assert_allclose(fr3.get_state()["qpos"][0], q0, atol=1e-6)

    fr3.reset()
    fr3.step(np.full((2, n), probe), 100)
    before = fr3.get_state()
    fr3.reset(np.array([0]))
    after = fr3.get_state()
    np.testing.assert_allclose(after["qpos"][0], q0, atol=1e-6)
    np.testing.assert_array_equal(after["qpos"][1], before["qpos"][1])
    np.testing.assert_array_equal(after["qvel"][1], before["qvel"][1])

    q_arb = q0 + np.linspace(0.1, -0.1, n)
    v_arb = np.linspace(0.05, -0.05, n)
    fr3.set_state(np.array([1]), q_arb[None, :], v_arb[None, :])
    rb = fr3.get_state()
    np.testing.assert_allclose(rb["qpos"][1], q_arb, atol=1e-6)
    np.testing.assert_allclose(rb["qvel"][1], v_arb, atol=1e-6)


def test_fr3_environment_isolation(fr3):
    n = len(EFFORT_LIMITS)
    kd = _kd(fr3)
    effort = np.asarray(EFFORT_LIMITS)
    q0 = fr3.get_default_qpos().copy()
    amplitudes = np.asarray(compare.sweep_amplitudes(
        compare.PROFILES["fr3_v2"], q0, fr3.get_joint_range()))
    single = create_backend(
        "superdex", SceneCfg(str(_bot_path())), 1, compare.SIM_DT,
        superdex_effort_limits=list(EFFORT_LIMITS))
    try:
        fr3.reset()
        single.reset()
        deviation = 0.0
        for k in range(60):
            target = _sweep_target(q0, amplitudes, k)
            tau_own = compare.pd_torque(KP, kd, effort, target,
                                        fr3.get_dof_pos()[0], fr3.get_dof_vel()[0])
            commands = np.zeros((2, n))
            commands[0] = tau_own
            commands[1] = compare.pd_torque(KP, kd, effort, q0,
                                            fr3.get_dof_pos()[1], fr3.get_dof_vel()[1])
            fr3.step(commands, compare.DECIMATION)
            tau_single = compare.pd_torque(KP, kd, effort, target,
                                           single.get_dof_pos()[0], single.get_dof_vel()[0])
            single.step(tau_single[None, :], compare.DECIMATION)
            deviation = max(deviation, float(np.max(np.abs(
                fr3.get_dof_pos()[0] - single.get_dof_pos()[0]))))
    finally:
        single.close()
    assert deviation <= TOL


def test_fr3_repeated_create_step_close_cycles():
    n = len(EFFORT_LIMITS)
    for _ in range(3):
        backend = create_backend(
            "superdex", SceneCfg(str(_bot_path())), 1, compare.SIM_DT,
            superdex_effort_limits=list(EFFORT_LIMITS))
        try:
            backend.step(np.zeros((1, n)), 10)
            backend.reset()
            np.testing.assert_allclose(
                backend.get_state()["qpos"][0], backend.get_default_qpos(), atol=1e-6)
        finally:
            backend.close()


def test_compare_tool_fr3_run_passes():
    """The consolidated comparison tool passes its full FR3 check set."""
    task = compare.WorkerTask(
        label="fr3_v2", path=str(_bot_path()), kind="bot",
        steps=compare.SWEEP_CTRL_STEPS * compare.DECIMATION)
    result = compare.run_model_subprocess(task, timeout=900)
    assert result["status"] == "passed", result.get("worker_output_tail", "")[-600:]
    assert set(result["checks"]) >= {
        "structure", "control", "trajectory_equivalence", "contact_recovery",
        "reset", "isolation", "lifecycle",
    }
