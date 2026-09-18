"""Native root metadata regressions and opt-in hand qualification."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.superdex.materialization import _native_plan
from unisim.backend.superdex.root_state import RootReference


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
        get_articulated_shape_info=lambda: SimpleNamespace(dof_info=[
            SimpleNamespace(offset=0, get_size=lambda: 6 if root == "FREE" else 0),
            SimpleNamespace(offset=6 if root == "FREE" else 0, get_size=lambda: 1),
        ]),
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
            HARD="HARD", FREE="FREE", REVOLUTE="REVOLUTE",
            PRISMATIC="PRISMATIC", SPHERICAL="SPHERICAL"
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
            get_articulated_actor=lambda: actor, get_sensor_handles=lambda: [],
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
    np.testing.assert_allclose(plan.default_qpos[:3], [1.2, -.3, .4])


def test_translated_rotated_root_frames_have_correct_origin_velocity():
    half = np.sqrt(.5)
    reference = RootReference(
        np.array([.2, -.3, .4]), np.array([half, 0, 0, half]),
        np.array([.1, 0, 0]), np.array([half, half, 0, 0]),
    )
    q = np.tile([.1, .2, .3, 1, 0, 0, 0], (2, 1))
    v = np.tile([1., 2, 3, 4, 5, 6], (2, 1))
    world_q, world_v = reference.to_world(q, v)
    np.testing.assert_allclose(world_q, np.tile([0, -.1, .7, .5, .5, .5, .5], (2, 1)),
                               atol=1e-14)
    np.testing.assert_allclose(world_v, np.tile([-2.6, 1, 2.5, 4, 6, -5], (2, 1)))
    back_q, back_v = reference.from_world(world_q, world_v)
    np.testing.assert_allclose(back_q, q, atol=1e-14)
    np.testing.assert_allclose(back_v, v, atol=1e-14)


@pytest.mark.parametrize("key", [
    "allegro_v5_left", "allegro_v5_right",
    "dg5f_short_left", "dg5f_short_right", "dg5f_long_left", "dg5f_long_right",
    "openarm_v20_left_gripper", "openarm_v20_right_gripper",
    "wuji_hand2_beta1_left", "wuji_hand2_beta1_right",
])
def test_floating_model_qualification(key):
    assets = os.environ.get("SUPERDEX_ASSETS_PATH")
    if not assets:
        pytest.skip("set SUPERDEX_ASSETS_PATH for local floating-hand qualification")
    pytest.importorskip("superdex.physics")
    script = Path(__file__).resolve().parents[1] / "scripts/superdex_floating_qualify.py"
    spec = importlib.util.spec_from_file_location("floating_qualify", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = module.discover_floating(Path(assets))
    result = module.qualify(Path(assets) / candidates[key])
    assert set(result) == {"serial", "batch"}


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
            translation=[.2, -.1, .3], rotation=p.Quaternion.from_rotation_vector([.4, .2, -.3])
        )
        cfg.links[0].parent_joint_from_link = p.TransformRT(
            translation=[.03, -.02, .04], rotation=p.Quaternion.from_rotation_vector([-.2, .3, .1])
        )
        return cfg

    monkeypatch.setattr(r, "load_bot_prefab_from_file", load_with_offsets)
    script = Path(__file__).resolve().parents[1] / "scripts/superdex_floating_qualify.py"
    spec = importlib.util.spec_from_file_location("floating_offset_qualify", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.qualify(Path(assets) / module.HANDS["allegro_v5_right"], steps=64)
    assert set(result) == {"serial", "batch"}


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

    monkeypatch.setitem(sys.modules, "superdex.physics.viewer", SimpleNamespace(
        VIEWER_AVAILABLE=True, Viewer=Viewer, ViewerCfg=lambda **kw: kw,
    ))
    backend = SimpleNamespace(_execution_mode="serial", num_envs=1, _worlds=[object()])

    def play():
        SuperDexBackend._run_interactive_playback(
            backend, env=None, initialize=lambda: event("initialize"),
            step=lambda obs: event("step"), num_steps=None, offscreen=False,
            debug_overlay_getter=None, on_frame=None,
        )

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            play()
    else:
        play()
        assert calls.count("step") == 1
    assert calls[-1] == "close"
    assert calls.count("close") == 1
