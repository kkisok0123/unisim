"""Shared boundary, SDK-independent application use, and native conversion checks."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from unisim import (
    ArticulationPoseTarget,
    CartesianTarget,
    FakeBackend,
    JointTarget,
    SimBackend,
    SuperDexBackend,
    create_backend,
)
from unisim.scene import SceneCfg

ROOT = Path(__file__).resolve().parents[1]


def test_shared_boundary_and_imports():
    for cls in SuperDexBackend.__mro__:
        if cls in (SimBackend, object):
            continue
        for name, value in vars(cls).items():
            if not name.startswith("_") and (callable(value) or isinstance(value, property)):
                assert hasattr(SimBackend, name), name
    code = (
        "import sys; from unisim import JointTarget, CartesianTarget, "
        "ArticulationPoseTarget, BackendModelInfo, BackendControllerInfo; "
        "assert not any(n.split('.')[0] in "
        "{'superdex','mujoco','torch','warp','genesis'} for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_optional_defaults_and_idempotent_cleanup():
    b = FakeBackend()
    for name, args in [
        ("get_model_info", ()),
        ("get_controller_descriptions", ()),
        ("set_gravity", ([0, 0, 0],)),
        ("get_camera_names", ()),
        ("get_camera_parameters", ("camera",)),
        ("get_camera_poses", ("camera",)),
        ("configure_controller", ("controller",)),
        ("clear_controller", ()),
        ("step_controller", ([JointTarget([0])],)),
    ]:
        with pytest.raises(NotImplementedError):
            getattr(b, name)(*args)
    handle = Mock()
    b._scene_cleanup_handle = handle
    b.close()
    b.close()
    handle.cleanup.assert_called_once()


@pytest.mark.parametrize("script", ["superdex_rigid_viewer.py", "superdex_bot_viewer.py"])
def test_normal_viewers_do_not_import_sdk_or_access_private_backend(script):
    tree = ast.parse((ROOT / "scripts" / script).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split(".")[0] != "superdex" for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "superdex"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "backend":
                assert not node.attr.startswith("_") and node.attr != "model"


@pytest.fixture
def assets(tmp_path):
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        pytest.skip("SuperDex wheels require Python 3.12/3.13")
    pytest.importorskip("superdex.physics")
    pytest.importorskip("superdex.robotics")
    sys.path.insert(0, str(ROOT / "scripts"))
    from superdex_rigid_fixtures import generate

    return generate(tmp_path)


def backend(path, **kwargs):
    return create_backend(
        "superdex", SceneCfg(str(path)), 1, 0.002, superdex_execution_mode="serial", **kwargs
    )


def test_metadata_scalar_limits_and_public_lifecycle(assets):
    b = backend(
        assets["multiple"],
        superdex_controlled_joints=["first/robot/a", "second/robot/b"],
        superdex_effort_limits=2.5,
    )
    try:
        info = b.get_model_info()
        assert len(info.articulations) == 2
        assert len(info.coordinate_names) == len(b.get_dof_pos()[0])
        assert "rotation_vector_component" in info.coordinate_representations
        assert info.body_articulation_ids[0] is None
        for i, a in enumerate(info.articulations):
            assert all(info.body_articulation_ids[j] == i for j in a.body_ids)
            assert info.body_names[a.root_body_id]
        groups = info.joint_coordinate_groups.copy()
        info.joint_coordinate_groups.clear()
        assert b.get_model_info().joint_coordinate_groups == groups
        np.testing.assert_array_equal(b.get_actuator_ctrl_range(), [[-2.5, 2.5]] * 4)
        assert b.get_joint_dof_pos_indices(["first/robot/a"]).size == 3
        state = b.get_state()
        assert state["qpos"].shape == (1, info.nq)
        b.step(np.zeros((1, b.num_actuators)), 10)
        b.set_state(np.array([0]), **state)
        b.reset()
        assert all(not c.available for c in b.get_controller_descriptions())
    finally:
        b.close()
        b.close()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, [1]])
def test_invalid_effort_limits(assets, value):
    with pytest.raises(ValueError, match="effort_limits"):
        backend(assets["plain"], superdex_effort_limits=value)


def test_passive_scalar_and_gravity_persists_after_reset(assets):
    b = backend(assets["spherical"], superdex_controlled_joints=[], superdex_effort_limits=1)
    try:
        info = b.get_model_info()
        assert b.num_actuators == 0
        root = info.articulations[0]
        assert root.floating
        b.set_gravity([0, 0, 0])
        initial = b.get_state()
        b.set_state(np.array([0]), initial["qpos"], np.zeros_like(initial["qvel"]))
        b.step(np.empty((1, 0)), 30)
        np.testing.assert_allclose(b.get_state()["qpos"], initial["qpos"], atol=1e-6)
        b.reset()
        np.testing.assert_array_equal(b.get_gravity(), [0, 0, 0])
        b.set_state(np.array([0]), initial["qpos"], np.zeros_like(initial["qvel"]))
        b.step(np.empty((1, 0)), 30)
        np.testing.assert_allclose(b.get_state()["qpos"], initial["qpos"], atol=1e-6)
        with pytest.raises(ValueError, match="gravity"):
            b.set_gravity([0, float("nan"), 0])
        np.testing.assert_array_equal(b.get_gravity(), [0, 0, 0])
        b.set_gravity([0, 0, -1])
        b.reset()
        b.step(np.empty((1, 0)), 30)
        slot = b.get_root_state_layout(info.body_names[root.root_body_id]).qvel_indices[2]
        assert b.get_state()["qvel"][0, slot] < -0.03
    finally:
        b.close()


def test_joint_targets_batch_validation_and_legacy_warning(assets):
    from superdex import robotics as r

    b = create_backend(
        "superdex", SceneCfg(str(assets["plain"])), 2, 0.002, superdex_execution_mode="serial"
    )
    try:
        descriptions = b.get_controller_descriptions()
        assert {d.target_kind for d in descriptions} == {"joint", "cartesian", "articulation_pose"}
        b.configure_controller(
            "BASIC_JSC_PD",
            param_args=json.dumps(
                {
                    "Kp": [5, 5],
                    "Kd": [1, 1],
                    "saturation": [1, 1],
                    "deadband": [0, 0],
                }
            ),
        )
        before = b.get_state()
        with pytest.raises(ValueError):
            b.step_controller([JointTarget([0.1, 0.1]), JointTarget([float("nan"), 0])])
        for k, v in before.items():
            np.testing.assert_array_equal(b.get_state()[k], v)
        with pytest.raises(TypeError, match="target kind"):
            b.step_controller([JointTarget([0, 0]), CartesianTarget([0, 0, 0, 1, 0, 0, 0])])
        with pytest.warns(DeprecationWarning, match="Native SuperDex"):
            b.step_controller(
                [JointTarget([0.1, -0.1]), r.ControllerBasicJscPdTarget(target_pose=[0.1, -0.1])],
                20,
            )
        np.testing.assert_array_equal(b.get_state()["qpos"][0], b.get_state()["qpos"][1])
    finally:
        b.close()


def test_pose_targets_and_legacy_conversion_are_equivalent(assets):
    from superdex import physics as p
    from superdex import robotics as r

    b = create_backend(
        "superdex", SceneCfg(str(assets["plain"])), 2, 0.002, superdex_execution_mode="serial"
    )
    try:
        b.configure_controller(
            "MOCHI_ARTICULATED_POSE",
            param_args=json.dumps(
                {
                    "poseControllerParams": {
                        "jointTracking": [{}, {"stiffness": 5}, {"stiffness": 5}]
                    }
                }
            ),
            init_args="{}",
        )
        root = [0, 0, 0, 1, 0, 0, 0]
        links = np.array([root, [0.3, 0, 0, 1, 0, 0, 0], [-0.3, 0, 0, 1, 0, 0, 0]])
        native = r.ControllerMochiArticulatedPoseTarget()
        native.world_from_root = p.TransformRT()
        native.local_to_parent_transforms = [
            p.TransformRT(translation=t[:3], rotation=t[[4, 5, 6, 3]]) for t in links
        ]
        target = ArticulationPoseTarget(root, link_poses=links)
        with pytest.warns(DeprecationWarning):
            b.step_controller([target, native], 20)
        np.testing.assert_array_equal(b.get_state()["qpos"][0], b.get_state()["qpos"][1])
        for invalid in [
            ArticulationPoseTarget(root),
            ArticulationPoseTarget(root, [0, 0], links),
            ArticulationPoseTarget([0] * 7, [0, 0]),
        ]:
            before = b.get_state()
            with pytest.raises(ValueError):
                b.step_controller([target, invalid])
            np.testing.assert_array_equal(b.get_state()["qpos"], before["qpos"])
    finally:
        b.close()


@pytest.mark.parametrize("mode", ["serial", "batch"])
def test_gravity_override_all_environments_and_floating_metadata(assets, mode):
    path = assets["plain"].parent / "free.superdex_bot"
    data = json.loads(assets["plain"].read_text())
    data["joints"][0]["type"] = "Free"
    data["defaultPose"] = [0, 0, 1, 0, 0, 0, 0, 0]
    path.write_text(json.dumps(data))
    b = create_backend(
        "superdex",
        SceneCfg(str(path)),
        2,
        0.002,
        superdex_execution_mode=mode,
        superdex_effort_limits=3,
    )
    try:
        info = b.get_model_info()
        assert info.articulations[0].floating
        assert info.coordinate_qpos_indices == (7, 8)
        assert info.coordinate_qvel_indices == (6, 7)
        osc = next(d for d in b.get_controller_descriptions() if d.identifier == "BASIC_OSC_PD")
        assert not osc.available and "SDK" in osc.unavailable_reason
        b.set_gravity([0, 0, -2])
        b.reset()
        b.step(np.zeros((2, 2)), 20)
        np.testing.assert_allclose(b.get_base_lin_vel()[:, 2], -0.08, atol=1e-5)
        b.set_gravity([0, 0, 0])
        b.reset(np.array([1]))
        b.step(np.zeros((2, 2)), 20)
        np.testing.assert_allclose(b.get_base_lin_vel()[:, 2], [-0.08, 0], atol=1e-5)
        b.reset()
        b.step(np.zeros((2, 2)), 20)
        np.testing.assert_allclose(b.get_base_lin_vel(), 0, atol=1e-5)
    finally:
        b.close()


def test_cartesian_target_quaternion_conversion(assets):
    from superdex import physics as p
    from superdex import robotics as r

    b = create_backend(
        "superdex", SceneCfg(str(assets["plain"])), 2, 0.002, superdex_execution_mode="serial"
    )
    try:
        b.configure_controller(
            "BASIC_OSC_PD",
            param_args=json.dumps(
                {
                    "Kp_p": 5,
                    "Kd_p": 1,
                    "Kp_r": 5,
                    "Kd_r": 1,
                    "bApplyMaxOSCTorqueNormalization": True,
                    "maxTranslationError": 1,
                    "maxRotationError": 1,
                }
            ),
            init_args='{"baseLinkName":"root","eeLinkName":"left"}',
        )
        pose = np.array([0.3, 0.01, 0, np.cos(0.05), 0, 0, np.sin(0.05)])
        native = r.ControllerBasicOscPdTarget(
            root_from_target_ee=p.TransformRT(
                translation=pose[:3],
                rotation=pose[[4, 5, 6, 3]],
            )
        )
        with pytest.warns(DeprecationWarning):
            b.step_controller([CartesianTarget(pose), native], 20)
        np.testing.assert_array_equal(b.get_state()["qpos"][0], b.get_state()["qpos"][1])
        before = b.get_state()
        with pytest.raises(ValueError, match="normalized"):
            b.step_controller([CartesianTarget(pose), CartesianTarget([0] * 7)])
        np.testing.assert_array_equal(b.get_state()["qpos"], before["qpos"])
    finally:
        b.close()


@pytest.mark.parametrize("mode", ["serial", "batch"])
def test_bot_placement_is_included_in_public_root_state(assets, mode):
    path = assets["plain"].parent / "placed.superdex_bot"
    data = json.loads(assets["plain"].read_text())
    data["joints"][0]["type"] = "Free"
    data["defaultPose"] = [0, 0, 1, 0, 0, 0, 0, 0]
    data["worldFromRoot"] = {
        "translation": [0.4, -0.2, 0.3],
        "rotation": [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
    }
    path.write_text(json.dumps(data))
    b = create_backend("superdex", SceneCfg(str(path)), 2, 0.002, superdex_execution_mode=mode)
    try:
        info = b.get_model_info()
        root = info.articulations[0].root_body_id

        def compare():
            state = b.get_state()
            np.testing.assert_allclose(
                state["qpos"][:, :3], b.get_body_pos_w([root])[:, 0], atol=2e-6
            )
            actual, expected = state["qpos"][:, 3:7], b.get_body_quat_w([root])[:, 0]
            np.testing.assert_allclose(np.abs(np.sum(actual * expected, axis=1)), 1, atol=2e-6)
            np.testing.assert_allclose(
                # SDK float32 body and generalized-velocity readbacks differ by a few ULPs.
                state["qvel"][:, :3], b.get_body_lin_vel_w([root])[:, 0], atol=1e-5
            )

        compare()
        b.step(np.zeros((2, 2)), 40)
        compare()
        state = b.get_state()
        b.set_state(np.array([0, 1]), **state)
        compare()
        b.reset()
        compare()
    finally:
        b.close()
