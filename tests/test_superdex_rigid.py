"""General rigid ownership, native joint layouts and portable serialized assets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.superdex.scenes import audit_scene
from unisim.scene import SceneCfg

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from superdex_compare import synthetic_fixtures as generate  # noqa: E402


@pytest.fixture
def fixtures(tmp_path):
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        pytest.skip("SuperDex wheels require Python 3.12 or 3.13")
    pytest.importorskip("superdex.physics")
    return generate(tmp_path)


def qualify(path, root, steps=1000):
    """Run the consolidated comparison tool's worker on one fixture."""
    import importlib

    compare = importlib.import_module("superdex_compare")
    task = compare.WorkerTask(label=path.stem, path=str(path), kind=compare._kind_of(path),
                              steps=steps)
    return compare.run_model_subprocess(task, timeout=600)


@pytest.mark.parametrize(
    "name",
    [
        "plain",
        "transmission",
        "tendon",
        "archive",
        "camera_archive",
        "ball",
        "spherical",
        "multiple",
        "nested",
        "contact",
        "external",
    ],
)
def test_native_rigid_sdk_equivalence_and_lifecycle(fixtures, name):
    result = qualify(fixtures[name], fixtures[name].parent, steps=1000)
    assert result["status"] == "passed", result.get("worker_output_tail", "")[-400:]
    # The adapter and the direct SDK integrate the same code path, so the
    # trajectory comparison is exact. Bots carry the lifecycle
    # create/step/reset/close equivalent of the scene cleanup check.
    trajectory = result["checks"].get("trajectory_equivalence", {})
    assert trajectory.get("passed"), trajectory
    recreation = result["checks"].get("cleanup_recreation") or result["checks"]["lifecycle"]
    assert recreation["passed"]
    if name == "contact":
        assert result["checks"]["trajectory_equivalence"][
            "reference_contact_force_steps"] > 0


def create(path, joints=None, limits=None):
    from unisim import create_backend

    return create_backend(
        "superdex",
        SceneCfg(str(path)),
        1,
        0.002,
        superdex_execution_mode="serial",
        superdex_controlled_joints=joints,
        superdex_effort_limits=limits,
    )


def test_multiactor_spherical_mapping_clipping_and_force_ownership(fixtures):
    path = fixtures["multiple"]
    b = create(path, ["first/robot/a", "second/robot/b"], [1, 2, 3, 4])
    reference = create(path, ["first/robot/a", "second/robot/b"], [1, 2, 3, 4])
    try:
        np.testing.assert_array_equal(b.get_joint_state_qpos_indices(["first/robot/a"]), [7, 8, 9])
        np.testing.assert_array_equal(b.get_joint_state_qvel_indices(["second/robot/b"]), [19])
        assert b.num_actuators == 4 and b.model.nq == 22 and b.model.nv == 20
        assert b.get_dof_armature().shape == (20,)
        np.testing.assert_allclose(b.get_joint_range()[:3], [[-0.4, 0.4]] * 3)
        b.step(np.array([[100, -100, 100, -100]]), 5)
        reference.step(np.array([[1, -2, 3, -4]]), 5)
        np.testing.assert_array_equal(b.get_state()["qvel"], reference.get_state()["qvel"])
        b.reset()
        reference.reset()
        body = b.get_body_id("first/robot/left")
        b.apply_body_force(np.array([body]), np.array([[[1.0, 2.0, 3.0]]]))
        b.step(np.zeros((1, 4)), 3)
        reference.step(np.zeros((1, 4)), 3)
        actual, expected = b.get_state(), reference.get_state()
        assert not np.array_equal(actual["qvel"][0, :10], expected["qvel"][0, :10])
        np.testing.assert_array_equal(actual["qvel"][0, 10:], expected["qvel"][0, 10:])
        np.testing.assert_array_equal(actual["qpos"][0, 11:], expected["qpos"][0, 11:])
        layout = b.get_root_state_layout("second/robot/root")
        assert layout.qpos_indices == tuple(range(11, 18))
        before = b.get_state()
        before["qpos"][0, 14:18] = [0, 0, 0, 0]
        with pytest.raises(ValueError, match="normalized"):
            b.set_state(np.array([0]), **before)
    finally:
        b.close()
        reference.close()


def test_overlapping_controls_and_new_batch_profiles_rejected(fixtures):
    with pytest.raises(ValueError, match="overlap"):
        create(fixtures["spherical"], ["a", "a/x"], [1] * 4)
    from unisim import create_backend

    with pytest.raises(NotImplementedError, match="serial"):
        create_backend(
            "superdex",
            SceneCfg(str(fixtures["multiple"])),
            1,
            0.002,
            superdex_execution_mode="batch",
            superdex_controlled_joints=[],
        )


def test_couplings_change_physics_without_adding_actions(fixtures):
    from unisim import create_backend

    states = []
    for name in ("plain", "transmission", "tendon"):
        b = create_backend(
            "superdex", SceneCfg(str(fixtures[name])), 1, 0.002, superdex_execution_mode="serial"
        )
        try:
            assert b.num_actuators == 2 and b.model.nv == 2
            b.step(np.zeros((1, 2)), 40)
            states.append(b.get_state()["qvel"])
        finally:
            b.close()
    assert not np.allclose(states[0], states[1])
    assert not np.allclose(states[0], states[2])


def test_general_scene_audit_keeps_deformables_and_unknown_fields_out(tmp_path):
    paths = generate(tmp_path)
    audit_scene(paths["multiple"], tmp_path)
    child = json.loads(paths["spherical"].read_text())
    child["actors"]["articulated"][0]["skin"] = {"shape": "./cube.obj", "typo": True}
    paths["spherical"].write_text(json.dumps(child))
    with pytest.raises(NotImplementedError, match="typo"):
        audit_scene(paths["multiple"], tmp_path)
    paths["spherical"].write_text(json.dumps({"actors": {"soft": []}}))
    with pytest.raises(NotImplementedError, match="soft"):
        audit_scene(paths["multiple"], tmp_path)


@pytest.mark.parametrize("kind", ["BASIC_JSC_PD", "MOCHI_ARTICULATED_POSE"])
def test_spherical_controller_targets_match_sdk(fixtures, kind):
    from superdex import physics as p
    from superdex import robotics as r

    from unisim import ArticulationPoseTarget, JointTarget, create_backend

    path = fixtures["ball"]
    b = create_backend("superdex", SceneCfg(str(path)), 1, 0.002, superdex_execution_mode="serial")
    world = p.create_scene("ball_controller_reference")
    context = r.create_context()
    bot = None
    try:
        world.set_gravity(b.get_gravity())
        bot = r.create_bot(world, r.load_bot_prefab_from_file(str(path)), context)
        actor = bot.get_articulated_actor()
        if kind == "BASIC_JSC_PD":
            params = {"Kp": [5] * 4, "Kd": [1] * 4, "saturation": [1] * 4, "deadband": [0] * 4}
            target = r.ControllerBasicJscPdTarget(target_pose=[0.1, 0.02, -0.03, 0.04])
        else:
            params = {
                "poseControllerParams": {
                    "jointTracking": [
                        {},
                        {"stiffness": 5, "damping": 1},
                        {"stiffness": 5, "damping": 1},
                    ]
                }
            }
            target = r.ControllerMochiArticulatedPoseTarget()
            target.world_from_root = p.TransformRT()
            target.pose_dofs = [0.1, 0.02, -0.03, 0.04]
        shared = (JointTarget([0.1, 0.02, -0.03, 0.04]) if kind == "BASIC_JSC_PD"
                  else ArticulationPoseTarget([0, 0, 0, 1, 0, 0, 0],
                                               [0.1, 0.02, -0.03, 0.04]))
        args = json.dumps(params)
        b.configure_controller(kind, param_args=args, init_args="{}")
        controller = bot.create_controller(kind)
        controller.configure_from_scene_entry(args, "{}")
        world.step(0)
        for _ in range(1000):
            b.step_controller([shared])
            if kind == "BASIC_JSC_PD":
                obsv = controller.get_current_observations_from_mochi()
                obsv.dt = 0.002
                forces = np.clip(np.asarray(controller.compute_output(obsv, target)), -1, 1)
            else:
                controller.compute_output(r.ControllerMochiArticulatedPoseObsv(), target)
                forces = np.zeros(4, dtype=np.float32)
            actor.set_external_forces_on_dofs(np.arange(4, dtype=np.int32), forces)
            world.step(0.002)
        expected = np.empty(4, dtype=np.float32)
        actor.get_articulated_pose(expected)
        np.testing.assert_allclose(b.get_state()["qpos"][0], expected, atol=1e-6)
        assert np.linalg.norm(expected[:3]) > 0.01
        b.reset()
        b.step_controller([shared], 3)
        assert np.isfinite(b.get_state()["qpos"]).all()
    finally:
        if bot is not None:
            if bot.get_articulated_actor().has_articulated_pose_controller():
                bot.get_articulated_actor().remove_articulated_pose_controller()
            r.destroy_bot(world, bot)
        p.destroy_scene(world)
        b.close()


def test_unnamed_instances_have_distinct_metadata_without_renaming_native_actors(fixtures):
    path = fixtures["multiple"]
    data = json.loads(path.read_text())
    for prefab in data["prefabs"]:
        prefab.pop("name")
    path.write_text(json.dumps(data))
    b = create(path, ["robot#0/a", "robot#1/b"], [1] * 4)
    try:
        assert len(set(b.model.body_names)) == len(b.model.body_names)
        assert [a.get_name() for a in b._actors[0].actors] == ["robot", "robot"]
        assert b.get_body_id("robot#1/root") != b.get_body_id("robot#0/root")
        b.step(np.zeros((1, 4)), 2)
    finally:
        b.close()


def test_joint_groups_are_derived_from_native_joint_identity():
    from types import SimpleNamespace

    from unisim.backend.superdex.model import coordinate_groups

    names = ("independent/x", "independent/y", "independent/z")
    groups = coordinate_groups(names, [0, 1, 2], [SimpleNamespace(name=n) for n in names])
    assert "independent" not in groups
    assert groups["independent/x"] == (0,)


def test_new_link_tracking_profile_requires_serial_even_for_scalar_joints(fixtures):
    from unisim import create_backend

    source = json.loads(fixtures["plain"].read_text())
    source.pop("defaultPose")
    for joint in source["joints"]:
        joint.pop("effortLimit")
    path = fixtures["plain"].with_suffix(".mochi_scene")
    path.write_text(
        json.dumps(
            {
                "actors": {"articulated": [source]},
                "controllers": [
                    {
                        "articulatedActor": "robot",
                        "linkPosTracking": [{"stiffness": 1, "damping": 0.1}],
                    },
                ],
            }
        )
    )
    with pytest.raises(NotImplementedError, match="serial"):
        create_backend("superdex", SceneCfg(str(path)), 1, 0.002, superdex_controlled_joints=[])
    b = create(path, [])
    try:
        b.step(np.zeros((1, 0)), 5)
        b.reset()
    finally:
        b.close()
