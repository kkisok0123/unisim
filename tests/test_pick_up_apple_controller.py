from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmark/pick_up_apple"


def _load(name: str, filename: str):
    path = BENCHMARK / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _joint(name: str) -> SimpleNamespace:
    kind = "REVOLUTE" if not name.endswith("hard") else "HARD"
    return SimpleNamespace(name=name, type=SimpleNamespace(name=kind))


def test_mujoco_gains_match_unchanged_superdex_controller() -> None:
    superdex_run = _load("pick_up_apple_superdex_run", "superdex/run.py")
    mujoco_runtime = _load("pick_up_apple_mujoco_runtime_gains", "mujoco_src/runtime.py")
    joints = [
        _joint("arm_openarm_right_joint1"),
        _joint("r_index_finger_mcp_flex"),
        _joint("r_middle_finger_dip"),
        _joint("l_index_finger_dip"),
        _joint("arm_to_right_hand_hard"),
    ]
    revolute = [joint.name for joint in joints if joint.type.name == "REVOLUTE"]

    normal = superdex_run.build_controller_params(joints, [], body_grasp=False)
    tracking = normal["poseControllerParams"]["jointTracking"]
    source_kp = np.asarray([row["stiffness"] for row in tracking[:-1]])
    source_kv = np.asarray([row["damping"] for row in tracking[:-1]])
    kp, kv = mujoco_runtime.build_position_gains(revolute)
    np.testing.assert_array_equal(kp, source_kp)
    np.testing.assert_array_equal(kv, source_kv)
    assert all(row["saturation"] == -1 for row in tracking)

    body = superdex_run.build_controller_params(joints, [], body_grasp=True)
    body_tracking = body["poseControllerParams"]["jointTracking"]
    source_kp = np.asarray([row["stiffness"] for row in body_tracking[:-1]])
    source_kv = np.asarray([row["damping"] for row in body_tracking[:-1]])
    kp, kv = mujoco_runtime.build_position_gains(revolute, body_grasp=True)
    np.testing.assert_array_equal(kp, source_kp)
    np.testing.assert_array_equal(kv, source_kv)
    assert all(row["saturation"] == -1 for row in body_tracking)


SCENE = BENCHMARK / "mujoco/scene.xml"


@pytest.fixture
def runtime():
    pytest.importorskip("mujoco")
    return _load("pick_up_apple_runtime_tests", "mujoco_src/runtime.py")


@pytest.fixture
def runner(monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(BENCHMARK / "mujoco_src"))
    return _load("pick_up_apple_runner_tests", "mujoco_src/run.py")


@pytest.mark.parametrize("motion", ["translation", "rotation", "stationary", "never_settles"])
def test_settling_requires_a_stable_pose_window(monkeypatch, motion, runner):
    class Backend:
        num_actuators = 0
        model = SimpleNamespace(jnt_qposadr=[0], jnt_dofadr=[0])
        steps = 0
        closed = False

        def materialize(self):
            pass

        def reset(self):
            pass

        def step(self, action, nsteps):
            self.steps += 1

        def get_state(self):
            t = self.steps * 0.1
            progress = t if motion == "never_settles" else min(t, 3.0)
            q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            if motion in ("translation", "never_settles"):
                q[0] = 0.01 * progress
            elif motion == "rotation":
                angle = 0.1 * progress
                q[3:5] = [np.cos(angle / 2), np.sin(angle / 2)]
            # Quaternion sign flips must not delay an otherwise stable pose.
            if self.steps % 2:
                q[3:] *= -1
            return {"qpos": q[None], "qvel": np.zeros((1, 6))}

        def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr(runner, "create_backend", lambda *a, **kw: backend)
    monkeypatch.setattr(runner.mujoco, "mj_name2id", lambda *a: 0)
    diagnostics = {}
    if motion == "never_settles":
        with pytest.raises(RuntimeError, match="did not sustain pose stability"):
            runner.settled_apple(SCENE, 0.1, diagnostics=diagnostics)
        assert backend.steps == 100
    else:
        runner.settled_apple(SCENE, 0.1, diagnostics=diagnostics)
        assert diagnostics["seconds"] == (3.0 if motion == "stationary" else 3.5)
    assert backend.closed


@pytest.mark.skipif(not SCENE.exists(), reason="generated MuJoCo scene is unavailable")
def test_real_apple_settles_before_planning(runtime, runner):
    path = runtime.build_settling_scene(SCENE.parent / "test_settling.xml")
    diagnostics = {}
    pose, velocity = runner.settled_apple(path, runtime.DT, diagnostics=diagnostics)
    assert diagnostics["seconds"] >= 3.0
    assert diagnostics["window_seconds"] >= 0.5
    assert diagnostics["position_excursion_m"] <= runner.SETTLE_POSITION_TOLERANCE
    assert diagnostics["angle_excursion_rad"] <= runner.SETTLE_ANGLE_TOLERANCE
    assert np.isfinite(pose).all() and np.isfinite(velocity).all()
    assert np.linalg.norm(velocity) > 0  # Residual motion is preserved, not overwritten.


def _load_isaac_runner():
    return _load("pick_up_apple_isaac_runner", "isaac/run.py")


def test_isaac_gain_schedule_matches_superdex_and_mujoco():
    pytest.importorskip("mujoco")
    superdex_run = _load("pick_up_apple_superdex_run", "superdex/run.py")
    isaac = _load_isaac_runner()
    mujoco_runtime = _load("pick_up_apple_mujoco_runtime", "mujoco_src/runtime.py")
    import xml.etree.ElementTree as ET

    root = ET.parse(SCENE).getroot()
    names = [item.get("name") for item in root.iter("joint") if item.get("type") == "hinge"]
    joints = [_joint(name) for name in names]

    for body_grasp in (False, True):
        source = superdex_run.build_controller_params(joints, [], body_grasp=body_grasp)[
            "poseControllerParams"
        ]["jointTracking"]
        expected_kp = np.asarray([row["stiffness"] for row in source])
        expected_kv = np.asarray([row["damping"] for row in source])
        actual_kp, actual_kv = isaac.build_position_gains(names, body_grasp=body_grasp)
        np.testing.assert_array_equal(actual_kp, expected_kp)
        np.testing.assert_array_equal(actual_kv, expected_kv)
        expected_mj_kp, expected_mj_kv = mujoco_runtime.build_position_gains(
            names, body_grasp=body_grasp
        )
        np.testing.assert_array_equal(actual_kp, expected_mj_kp)
        np.testing.assert_array_equal(actual_kv, expected_mj_kv)


def test_isaac_converts_per_radian_gains_to_per_degree_drives():
    isaac = _load_isaac_runner()
    kp = np.asarray([1000.0, 30.0, 0.8])
    kv = np.asarray([40.0, 0.02, 0.02])
    drive_kp, drive_kv = isaac.isaac_drive_gains(kp, kv)
    np.testing.assert_allclose(drive_kp, kp * np.pi / 180.0, rtol=0, atol=np.finfo(float).eps)
    np.testing.assert_allclose(drive_kv, kv * np.pi / 180.0, rtol=0, atol=np.finfo(float).eps)


def test_isaac_tensor_boundary_never_passes_numpy_to_articulation_apis(monkeypatch):
    pytest.importorskip("torch")
    isaac = _load_isaac_runner()

    calls = []

    class TorchCheck:
        @staticmethod
        def _check(name, value):
            assert not isinstance(value, np.ndarray), f"{name} received a NumPy array"
            np.testing.assert_allclose(value.detach().cpu().numpy(), getattr(TorchCheck, name))

        def set_joint_positions(self, positions, joint_indices):
            self._check("positions", positions)
            self._check("indices", joint_indices)
            calls.append("positions")

        def set_joint_velocities(self, velocities, joint_indices):
            self._check("velocities", velocities)
            self._check("indices", joint_indices)
            calls.append("velocities")

        positions = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        velocities = np.zeros(3, dtype=np.float32)
        indices = np.asarray([0, 1, 2], dtype=np.int32)

    robot = TorchCheck()
    joint_indices = isaac.to_torch(np.arange(3, dtype=np.int32), integer=True)
    robot.set_joint_positions(isaac.to_torch(TorchCheck.positions), joint_indices=joint_indices)
    robot.set_joint_velocities(isaac.to_torch(TorchCheck.velocities), joint_indices=joint_indices)
    assert calls == ["positions", "velocities"]


def test_isaac_viewer_clock_clamps_and_skips_headless_updates():
    isaac = _load_isaac_runner()

    class App:
        updates = 0

        def is_running(self):
            return True

        def update(self):
            self.updates += 1

    app = App()
    clock = isaac.ViewerClock(enabled=True, rate_hz=0.1, app=app)
    assert clock.rate_hz == 1.0
    clock = isaac.ViewerClock(enabled=True, rate_hz=120.0, app=app)
    assert clock.rate_hz == 60.0
    clock.service(0)
    assert app.updates == 1
    # The rate limiter suppresses the immediately following service call.
    clock.service(1)
    assert app.updates == 1
    headless = isaac.ViewerClock(enabled=False, rate_hz=20, app=app)
    headless.service(100)
    assert app.updates == 1
