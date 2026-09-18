"""Native root metadata regressions and opt-in hand qualification."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.superdex.materialization import _native_plan
from unisim.backend.superdex.model import RootReference


def fixture_runtime(root="FREE", child="REVOLUTE", offset=False):
    identity = SimpleNamespace(translation=np.zeros(3), rotation=np.array([0, 0, 0, 1]))
    transform = (
        SimpleNamespace(translation=np.array([1, 0, 0]), rotation=identity.rotation)
        if offset
        else identity
    )
    joints = [
        SimpleNamespace(type=root, parent_link_from_joint=transform, name="root"),
        SimpleNamespace(
            type=child,
            name="finger",
            axis=np.array([0.0, 0.0, 1.0]),
            min_limit=None,
            max_limit=None,
            effort_limit=2.0,
            inertia=0.01,
        ),
    ]
    links = [
        SimpleNamespace(
            name=name,
            parent_link=i - 1,
            sensors=[],
            actuators=[],
            mass=1.0,
            shape_file=None,
            parent_joint_from_link=identity,
        )
        for i, name in enumerate(("palm", "finger"))
    ]
    cfg = SimpleNamespace(
        joints=joints, links=links, cycles=[], linear_transmissions=[], spatial_tendons=[]
    )
    pose = np.array([0.2, -0.3, 0.4, 0, 0, 0, 0.1]) if root == "FREE" else np.array([0.1])
    actor = SimpleNamespace(
        get_root_transform=lambda: identity,
        get_nested_link_actors=lambda: [0, 1],
        get_articulated_pose=lambda out: np.copyto(out, pose),
        get_num_dofs=lambda: len(pose),
        get_articulated_shape_info=lambda: SimpleNamespace(
            dof_info=[
                SimpleNamespace(offset=0, get_size=lambda: 6 if root == "FREE" else 0),
                SimpleNamespace(offset=6 if root == "FREE" else 0, get_size=lambda: 1),
            ]
        ),
    )
    link = SimpleNamespace(
        is_static=lambda: False,
        get_mass=lambda: 1.0,
        get_root_transform=lambda: identity,
        get_center_of_mass_transform=lambda: identity,
    )
    scene = SimpleNamespace(get_actor=lambda h: link)
    physics = SimpleNamespace(
        ArticulatedJointType=SimpleNamespace(
            HARD="HARD",
            FREE="FREE",
            REVOLUTE="REVOLUTE",
            PRISMATIC="PRISMATIC",
            SPHERICAL="SPHERICAL",
        ),
        create_scene=lambda name: scene,
        destroy_scene=lambda s: None,
        uses_double_precision=lambda: False,
        Quaternion=SimpleNamespace(from_rotation_vector=lambda v: identity.rotation),
    )
    robotics = SimpleNamespace(
        load_bot_prefab_from_file=lambda path: cfg,
        create_context=lambda: object(),
        create_bot=lambda *args: SimpleNamespace(
            get_articulated_actor=lambda: actor,
            get_sensor_handles=lambda: [],
            get_actuator_handles=lambda: [],
        ),
        destroy_bot=lambda *args: None,
    )
    return physics, robotics


@pytest.mark.parametrize("root,qo,vo", [("FREE", 7, 6), ("HARD", 0, 0)])
def test_native_root_dimensions_and_control_offsets(root, qo, vo):
    p, r = fixture_runtime(root)
    plan = _native_plan(p, r, Path("synthetic.superdex_bot"), None)
    assert (plan.nq, plan.nv) == (qo + 1, vo + 1)
    assert plan.joint_names == ("finger",)
    assert plan.actuator_joint_names == ("finger",)
    np.testing.assert_array_equal(plan.joint_qpos_indices, [qo])
    np.testing.assert_array_equal(plan.actuator_qvel_indices, [vo])
    np.testing.assert_allclose(plan.dof_armature, [0.0] * vo + [0.01])
    np.testing.assert_allclose(plan.default_qpos[-1], 0.1)
    if qo:
        np.testing.assert_allclose(plan.default_qpos[:7], [0.2, -0.3, 0.4, 1, 0, 0, 0])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"child": "FREE"}, "fixed/hinge/slide"),
        ({"root": "SPHERICAL"}, "HARD or FREE"),
    ],
)
def test_unsupported_native_root_profiles_are_rejected(kwargs, match):
    p, r = fixture_runtime(**kwargs)
    with pytest.raises(NotImplementedError, match=match):
        _native_plan(p, r, Path("synthetic.superdex_bot"), None)


def test_authored_root_translation_changes_default_world_pose():
    p, r = fixture_runtime(offset=True)
    plan = _native_plan(p, r, Path("synthetic.superdex_bot"), None)
    np.testing.assert_allclose(plan.default_qpos[:3], [1.2, -0.3, 0.4])


def test_translated_rotated_root_frames_have_correct_origin_velocity():
    half = np.sqrt(0.5)
    reference = RootReference(
        np.array([0.2, -0.3, 0.4]),
        np.array([half, 0, 0, half]),
        np.array([0.1, 0, 0]),
        np.array([half, half, 0, 0]),
    )
    q = np.tile([0.1, 0.2, 0.3, 1, 0, 0, 0], (2, 1))
    v = np.tile([1.0, 2, 3, 4, 5, 6], (2, 1))
    world_q, world_v = reference.to_world(q, v)
    np.testing.assert_allclose(
        world_q, np.tile([0, -0.1, 0.7, 0.5, 0.5, 0.5, 0.5], (2, 1)), atol=1e-14
    )
    np.testing.assert_allclose(world_v, np.tile([-2.6, 1, 2.5, 4, 6, -5], (2, 1)))
    back_q, back_v = reference.from_world(world_q, world_v)
    np.testing.assert_allclose(back_q, q, atol=1e-14)
    np.testing.assert_allclose(back_v, v, atol=1e-14)


FLOATING_HANDS = {
    "allegro_v5_left": "bots/hands/allegro_v5/left/allegro_v5_left.superdex_bot",
    "allegro_v5_right": "bots/hands/allegro_v5/right/allegro_v5_right.superdex_bot",
    "dg5f_short_left": "bots/hands/dg5f_short/left/dg5f_short_left.superdex_bot",
    "dg5f_short_right": "bots/hands/dg5f_short/right/dg5f_short_right.superdex_bot",
    "dg5f_long_left": "bots/hands/dg5f_long/left/dg5f_long_left.superdex_bot",
    "dg5f_long_right": "bots/hands/dg5f_long/right/dg5f_long_right.superdex_bot",
    "openarm_v20_left_gripper": (
        "bots/grippers/openarm_v20/left/openarm_v20_left_gripper.superdex_bot"
    ),
    "openarm_v20_right_gripper": (
        "bots/grippers/openarm_v20/right/openarm_v20_right_gripper.superdex_bot"
    ),
    "wuji_hand2_beta1_left": "bots/hands/wuji_hand2_beta1/left/wuji_hand2_beta1_left.superdex_bot",
    "wuji_hand2_beta1_right": (
        "bots/hands/wuji_hand2_beta1/right/wuji_hand2_beta1_right.superdex_bot"
    ),
}


@pytest.mark.parametrize("key", sorted(FLOATING_HANDS))
def test_floating_model_qualification(key):
    assets = os.environ.get("SUPERDEX_ASSETS_PATH")
    if not assets:
        pytest.skip("set SUPERDEX_ASSETS_PATH for local floating-hand qualification")
    pytest.importorskip("superdex.physics")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import superdex_compare as compare

    task = compare.WorkerTask(
        label=key,
        path=str(Path(assets) / FLOATING_HANDS[key]),
        kind="bot",
        steps=compare.FLOATING_STEPS,
    )
    result = compare.run_model_subprocess(task, timeout=900)
    assert result["status"] == "passed", result.get("worker_output_tail", "")[-400:]
    assert result["checks"]["trajectory_equivalence"]["passed"]


def test_native_translated_rotated_reference_frames(monkeypatch):
    """Exercise both authored transforms without modifying local asset bytes."""
    assets = os.environ.get("SUPERDEX_ASSETS_PATH")
    if not assets:
        pytest.skip("set SUPERDEX_ASSETS_PATH for authored-frame qualification")
    p = pytest.importorskip("superdex.physics")
    r = pytest.importorskip("superdex.robotics")
    load = r.load_bot_prefab_from_file

    def load_with_offsets(path):
        cfg = load(path)
        cfg.joints[0].parent_link_from_joint = p.TransformRT(
            translation=[0.2, -0.1, 0.3],
            rotation=p.Quaternion.from_rotation_vector([0.4, 0.2, -0.3]),
        )
        cfg.links[0].parent_joint_from_link = p.TransformRT(
            translation=[0.03, -0.02, 0.04],
            rotation=p.Quaternion.from_rotation_vector([-0.2, 0.3, 0.1]),
        )
        return cfg

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import superdex_compare as compare

    monkeypatch.setattr(r, "load_bot_prefab_from_file", load_with_offsets)
    path = Path(assets) / "bots/hands/allegro_v5/right/allegro_v5_right.superdex_bot"
    from unisim.backend.superdex.runtime import acquire_runtime, release_runtime

    acquire_runtime(p)
    try:
        run = compare.CheckRun("allegro_offset", "bot")
        compare.check_bot(compare.ModelRef(path, "bot", "allegro_offset"), run, steps=64 * 8)
        assert not run.failures, run.checks
    finally:
        release_runtime(p)


@pytest.mark.parametrize(
    "failure", [None, "set_scene", "frame_scene", "initialize", "step", "render"]
)
def test_viewer_closes_on_window_exit_and_errors(monkeypatch, failure):
    """Exercise ownership without importing the optional SDK or opening a window."""
    from unisim.backend.superdex.backend import SuperDexBackend

    calls = []

    def event(name):
        calls.append(name)
        if name == failure:
            raise RuntimeError(name)

    class Viewer:
        def __init__(self, cfg):
            pass

        def set_scene(self, scene):
            event("set_scene")

        def frame_scene(self):
            event("frame_scene")

        def render(self):
            event("render")

        def user_requested_close(self):
            return True

        def close(self):
            event("close")

    monkeypatch.setitem(
        sys.modules,
        "superdex.physics.viewer",
        SimpleNamespace(
            VIEWER_AVAILABLE=True,
            Viewer=Viewer,
            ViewerCfg=lambda **kw: kw,
        ),
    )
    backend = SimpleNamespace(_execution_mode="serial", num_envs=1, _worlds=[object()])

    def play():
        SuperDexBackend._run_interactive_playback(
            backend,
            env=None,
            initialize=lambda: event("initialize"),
            step=lambda obs: event("step"),
            num_steps=None,
            offscreen=False,
            debug_overlay_getter=None,
            on_frame=None,
        )

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            play()
    else:
        play()
        assert calls.count("step") == 1
    assert calls[-1] == "close"
    assert calls.count("close") == 1
