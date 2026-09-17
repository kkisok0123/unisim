"""Built-in component audits and optional native camera/controller qualification."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.superdex.components import audit_components

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/superdex_component_qualify.py"
spec = importlib.util.spec_from_file_location("component_qualify", SCRIPT)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


@pytest.mark.parametrize(
    "kind,field",
    [
        ("CUSTOM_POSITION_SERVO", "actuators"),
        ("CUSTOM_CONTACT_FORCE_SENSOR", "sensors"),
        ("OTHER", "actuators"),
        ("OTHER", "sensors"),
    ],
)
def test_custom_components_rejected_without_sdk(kind, field):
    link = SimpleNamespace(actuators=[], sensors=[])
    getattr(link, field).append(SimpleNamespace(name="custom", type=kind))
    with pytest.raises(NotImplementedError, match=kind):
        audit_components([link])
    with pytest.raises(NotImplementedError, match=kind):
        qualify.audit(SimpleNamespace(links=[link]))


def test_component_free_bots_and_explicit_cli_selection():
    assert audit_components([SimpleNamespace(actuators=[], sensors=[])]) == ()
    with pytest.raises(SystemExit) as exc:
        qualify.main([])
    assert exc.value.code == 2


def test_visualize_dispatches_two_live_viewers_without_writing_report(tmp_path, monkeypatch):
    bot = "bots/arms/fr3_v2/fr3_v2.superdex_bot"
    config_path = tmp_path / "jsc.json"
    target_path = tmp_path / "target.json"
    config_path.write_text(
        json.dumps({"type_name": "BASIC_JSC_PD", "param_args": "", "init_args": ""})
    )
    target_path.write_text(json.dumps({"type_name": "BASIC_JSC_PD", "target_pose": [0.0]}))
    monkeypatch.setattr(qualify, "resolve_assets_root", lambda value: tmp_path)
    monkeypatch.setattr(qualify, "discover_bots", lambda root: [bot])
    called = {}

    def run_live(relpath, root, fragments, limits, config, target):
        called.update(
            relpath=relpath,
            root=root,
            fragments=fragments,
            limits=limits,
            config=config,
            target=target,
        )
        return 0

    monkeypatch.setattr(qualify, "run_live_comparison", run_live)
    result = qualify.main(
        [
            "--bots",
            bot,
            "--effort-limit",
            "1",
            "--controller-config",
            str(config_path),
            "--absolute-target",
            str(target_path),
            "--visualize",
            "--skip-verification",
        ]
    )
    assert result == 0
    assert called == {
        "relpath": bot,
        "root": tmp_path,
        "fragments": None,
        "limits": [1.0],
        "config": {"type_name": "BASIC_JSC_PD", "param_args": "", "init_args": ""},
        "target": {"type_name": "BASIC_JSC_PD", "target_pose": [0.0]},
    }
    assert not list(tmp_path.glob("*.gif"))
    assert not list(tmp_path.glob("*.report.json"))


def test_camera_mount_and_names_are_validated_without_sdk():
    camera = SimpleNamespace(
        name="camera",
        type="SENSOR_CAMERA",
        params="{}",
        parent_from_sensor=SimpleNamespace(
            translation=np.zeros(3), rotation=np.array([0, 0, 0, 1.0])
        ),
    )
    link = SimpleNamespace(actuators=[], sensors=[camera])
    assert audit_components([link])[0].name == "camera"
    with pytest.raises(ValueError, match="unique"):
        audit_components([link, link])
    camera.parent_from_sensor.rotation[:] = 0
    with pytest.raises(ValueError, match="normalized"):
        audit_components([link])


def test_absolute_controller_targets_without_sdk():
    class ValueTarget:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class PoseTarget:
        pass

    class Transform:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    runtime = SimpleNamespace(
        ControllerBasicJscPdTarget=ValueTarget,
        ControllerBasicOscPdTarget=ValueTarget,
        ControllerMochiArticulatedPoseTarget=PoseTarget,
    )
    physics = SimpleNamespace(TransformRT=Transform)
    reference = SimpleNamespace(r=runtime, p=physics, nv=2, n=2, dtype=np.float32)

    reference.kind = "BASIC_JSC_PD"
    target = qualify.absolute_target(
        reference, {"type_name": reference.kind, "target_pose": [0.2, -0.4]}
    )
    np.testing.assert_array_equal(target.target_pose, np.array([0.2, -0.4], np.float32))

    reference.kind = "BASIC_OSC_PD"
    target = qualify.absolute_target(
        reference,
        {
            "type_name": reference.kind,
            "root_from_target_ee": {
                "translation": [0.1, 0.2, 0.3],
                "rotation_xyzw": [0, 0, 0, 1],
            },
        },
    )
    np.testing.assert_array_equal(target.root_from_target_ee.translation, [0.1, 0.2, 0.3])

    reference.kind = "MOCHI_ARTICULATED_POSE"
    target = qualify.absolute_target(
        reference,
        {
            "type_name": reference.kind,
            "world_from_root": {
                "translation": [0, 0, 0],
                "rotation_xyzw": [0, 0, 0, 1],
            },
            "pose_dofs": [0.2, -0.4],
        },
    )
    np.testing.assert_array_equal(target.pose_dofs, np.array([0.2, -0.4], np.float32))
    with pytest.raises(ValueError, match="normalized"):
        qualify.absolute_target(
            reference,
            {
                "type_name": reference.kind,
                "world_from_root": {
                    "translation": [0, 0, 0],
                    "rotation_xyzw": [0, 0, 0, 0],
                },
                "pose_dofs": [0.2, -0.4],
            },
        )


@pytest.fixture
def native(tmp_path, request):
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        pytest.skip("SuperDex runtime requires Python 3.12/3.13")
    r = pytest.importorskip("superdex.robotics")
    p = pytest.importorskip("superdex.physics")
    params = r.CameraSensorParams()
    params.image_width, params.image_height = 320, 240
    params.fov_vertical_deg, params.near_clip, params.far_clip = 60, 0.01, 100
    params.save_to_file(str(tmp_path / "camera.superdex_sensor"))
    floating = getattr(request, "param", False)
    data = {
        "name": "fixture",
        "defaultPose": ([0, 0, 0.5, 0, 0, 0] if floating else []) + [0.1],
        "joints": [
            {"name": "root", "type": "Free" if floating else "Hard"},
            {
                "name": "hinge",
                "type": "Revolute",
                "axis": [0, 0, 1],
                "parentLinkFromJoint": {"translation": [0, 0, 0.3]},
                "effortLimit": 2,
                "inertia": 0.01,
                "minLimit": [0, 0, -1],
                "maxLimit": [0, 0, 1],
            },
        ],
        "links": [
            {
                "name": "base",
                "mass": 1,
                "centerOfMass": [0, 0, 0],
                "momentOfInertia": [1, 0, 0, 1, 0, 1],
            },
            {
                "name": "tip",
                "parentLink": 0,
                "mass": 1,
                "centerOfMass": [0, 0, 0],
                "momentOfInertia": [1, 0, 0, 1, 0, 1],
                "sensors": [
                    {
                        "name": "camera",
                        "type": "SENSOR_CAMERA",
                        "params": "./camera.superdex_sensor",
                        "parentFromSensor": {
                            "translation": [0.1, 0.2, 0.3],
                            "rotation": [0, 0, 0.70710678, 0.70710678],
                        },
                    }
                ],
            },
        ],
    }
    path = tmp_path / "fixture.superdex_bot"
    path.write_text(json.dumps(data))
    # A closed tetrahedron gives the native floating root nonzero mass/inertia.
    # Shape-free SDK links have zero mass regardless of authored mass settings.
    (tmp_path / "shape.obj").write_text(
        "v 0 0 0\nv .05 0 0\nv 0 .05 0\nv 0 0 .05\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    for link in data["links"]:
        link["shape"] = "./shape.obj"
    path.write_text(json.dumps(data))
    return p, r, path, floating


def controller_config(r, p, kind, floating):
    if kind == "BASIC_JSC_PD":
        return dict(
            type_name=kind,
            param_args=json.dumps(
                {
                    "Kp": [0.0] * (6 if floating else 0) + [10.0],
                    "Kd": [0.0] * (6 if floating else 0) + [1.0],
                    "saturation": [2.0] * (7 if floating else 1),
                    "deadband": [0.0] * (7 if floating else 1),
                }
            ),
            init_args="",
        )
    if kind == "BASIC_OSC_PD":
        return dict(
            type_name=kind,
            param_args=json.dumps(
                {
                    "Kp_p": 10,
                    "Kd_p": 1,
                    "Kp_r": 10,
                    "Kd_r": 1,
                    "bApplyMaxOSCTorqueNormalization": not floating,
                    "maxTranslationError": 1,
                    "maxRotationError": 1,
                }
            ),
            init_args='{"baseLinkName":"base","eeLinkName":"tip"}',
        )
    # Serialize with the SDK so the test follows the public schema exactly.
    return dict(
        type_name=kind,
        param_args=json.dumps(
            {
                "poseControllerParams": {
                    "jointTracking": [
                        {"stiffness": 0, "damping": 0},
                        {"stiffness": 10, "damping": 1},
                    ]
                }
            }
        ),
        init_args="{}",
    )


@pytest.mark.parametrize("native", [False, True], indirect=True)
@pytest.mark.parametrize("kind", [None, "BASIC_JSC_PD", "BASIC_OSC_PD", "MOCHI_ARTICULATED_POSE"])
def test_universal_qualification_against_direct_sdk(native, kind, tmp_path):
    p, r, path, floating = native
    config = controller_config(r, p, kind, floating) if kind else None
    if floating and kind == "BASIC_OSC_PD":
        # Verify the SDK limitation independently, then require an explicit adapter diagnostic.
        from unisim import create_backend
        from unisim.scene import SceneCfg

        b = create_backend("superdex", SceneCfg(str(path)), 1, 0.002)
        try:
            reference = qualify.Reference(path, [], path.parent)
            try:
                with pytest.raises(Exception, match="bot-space vs actor-space"):
                    reference.configure(config)
            finally:
                reference.close()
            with pytest.raises(NotImplementedError, match="indexing mismatch"):
                b.configure_controller(**config)
            assert not b._controllers
            b.step(np.zeros((1, 1)))
        finally:
            b.close()
        return
    assert (
        qualify.qualify_bot(path.name, path.parent, tmp_path / "reports", controller_config=config)
        == 0
    )
    report = json.loads((tmp_path / "reports/fixture.json").read_text())
    assert report["checks"]["cameras"]["passed"]
    assert report["checks"]["trajectory_equivalence"]["physics_steps"] == 1040


@pytest.mark.parametrize(
    "kind,target_spec",
    [
        ("BASIC_JSC_PD", {"type_name": "BASIC_JSC_PD", "target_pose": [0.2]}),
        (
            "BASIC_OSC_PD",
            {
                "type_name": "BASIC_OSC_PD",
                "root_from_target_ee": {
                    "translation": [0.0, 0.0, 0.3],
                    "rotation_xyzw": [0, 0, 0, 1],
                },
            },
        ),
        (
            "MOCHI_ARTICULATED_POSE",
            {
                "type_name": "MOCHI_ARTICULATED_POSE",
                "world_from_root": {
                    "translation": [0, 0, 0],
                    "rotation_xyzw": [0, 0, 0, 1],
                },
                "pose_dofs": [0.2],
            },
        ),
    ],
)
def test_absolute_controller_target_qualification(native, kind, target_spec, tmp_path):
    p, r, path, _ = native
    config = controller_config(r, p, kind, False)
    assert (
        qualify.qualify_bot(
            path.name,
            path.parent,
            tmp_path / "reports",
            controller_config=config,
            absolute_target_spec=target_spec,
        )
        == 0
    )
    report = json.loads((tmp_path / "reports/fixture.json").read_text())
    assert report["absolute_target"] == target_spec
    assert report["checks"]["trajectory_equivalence"]["native_state_max_dev"] == 0


def test_native_live_worker_steps_with_list_limits(native, monkeypatch):
    from queue import Queue
    from threading import Barrier, Event

    p, r, path, _ = native
    config = controller_config(r, p, "BASIC_JSC_PD", False)
    calls = []
    viewer = SimpleNamespace(
        render=lambda: calls.append("render"),
        user_requested_close=lambda: True,
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(qualify, "_open_viewer", lambda scene, title: viewer)
    monkeypatch.setattr(qualify.signal, "signal", lambda *args: None)
    original_step = qualify.Reference.step

    def step(reference, target, limits):
        original_step(reference, target, limits)
        calls.append("step")

    monkeypatch.setattr(qualify.Reference, "step", step)
    ready = Queue()
    start, stop = Event(), Event()
    start.set()
    qualify._run_native_viewer(
        str(path),
        [],
        str(path.parent),
        [2.0],
        config,
        {"type_name": "BASIC_JSC_PD", "target_pose": [0.2]},
        ready,
        start,
        stop,
        Barrier(1),
    )
    assert calls == ["step"] * qualify.DECIMATION + ["render", "close"]
    assert ready.get_nowait() == ("Native SuperDex SDK", "ready", "")
    assert stop.is_set()
    assert not p.is_initialized()


@pytest.mark.parametrize("mode", ["batch", "serial"])
def test_camera_state_and_controller_lifecycle(native, mode):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    p, r, path, _ = native
    b = create_backend("superdex", SceneCfg(str(path)), 2, 0.002, superdex_execution_mode=mode)
    try:
        initial = b.get_camera_poses("camera")
        params = b.get_camera_parameters("camera")
        params["offset_local"][0] = 900
        assert b.get_camera_parameters("camera")["offset_local"][0] == 0
        with pytest.raises(ValueError, match="get_camera"):
            b.get_sensor_data("camera")
        q = b.get_state()["qpos"]
        q[0, 0] += 0.2
        b.set_state(np.array([0]), q[:1], np.zeros((1, 1)))
        assert not np.array_equal(b.get_camera_poses("camera")[0], initial[0])
        np.testing.assert_array_equal(b.get_camera_poses("camera", np.array([1]))[0], initial[1])
        b.reset()
        np.testing.assert_allclose(b.get_camera_poses("camera"), initial, atol=1e-7)
        for kind in ("BASIC_JSC_PD", "BASIC_OSC_PD", "MOCHI_ARTICULATED_POSE"):
            b.configure_controller(**controller_config(r, p, kind, False))
            with pytest.raises(RuntimeError, match="clear_controller"):
                b.step(np.zeros((2, 1)))
            with pytest.raises(RuntimeError, match="clear_controller"):
                b.configure_controller("BASIC_JSC_PD")
            with pytest.raises(RuntimeError, match="pre-step"):
                b.set_pre_step_control(lambda backend, ctrl: ctrl)
            before = b.get_state()["qpos"].copy()
            with pytest.raises((TypeError, ValueError)):
                b.step_controller([None, None])
            np.testing.assert_array_equal(before, b.get_state()["qpos"])
            b.clear_controller()
            assert not any(a.has_articulated_pose_controller() for a in b._actors)
            b.step(np.zeros((2, 1)))
        # Invalid OSC initialization releases both robotics and solver ownership.
        with pytest.raises(Exception):
            b.configure_controller("BASIC_OSC_PD", init_args='{"eeLinkName":"missing"}')
        assert not b._controllers
        for _, bot, _ in b.model.native_bots.values():
            assert len(bot.get_controller_handles()) == 0
        b.configure_controller(**controller_config(r, p, "BASIC_JSC_PD", False))
        valid = r.ControllerBasicJscPdTarget(target_pose=[0.2])
        invalid = r.ControllerBasicJscPdTarget(target_pose=[float("nan")])
        before = b.get_state()["qpos"].copy()
        with pytest.raises(ValueError, match="finite"):
            b.step_controller([valid, invalid])
        np.testing.assert_array_equal(before, b.get_state()["qpos"])
        b.clear_controller()
        b.set_pre_step_control(lambda backend, ctrl: ctrl)
        with pytest.raises(RuntimeError, match="pre-step"):
            b.configure_controller("BASIC_JSC_PD")
    finally:
        b.close()
        b.close()


def test_pose_controller_link_transform_targets(native):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    p, r, path, _ = native
    b = create_backend("superdex", SceneCfg(str(path)), 1, 0.002)
    reference = qualify.Reference(path, [], path.parent)
    try:
        config = controller_config(r, p, "MOCHI_ARTICULATED_POSE", False)
        b.configure_controller(**config)
        reference.configure(config)
        reference.seed(*qualify._native_state(b))
        poses = [link.get_root_transform() for link in b._links[0]]
        t = r.ControllerMochiArticulatedPoseTarget()
        t.world_from_root = poses[0]
        t.local_to_parent_transforms = [p.TransformRT(), poses[0].inverse() * poses[1]]
        b.step_controller([t], 4)
        for _ in range(4):
            reference.step(t, np.array([2.0]))
        for left, right in zip(qualify._native_state(b), reference.state()):
            np.testing.assert_array_equal(left, right)
        assert np.isfinite(b.get_state()["qpos"]).all()
        b.reset()
        b.step_controller([t])
    finally:
        reference.close()
        b.close()


def test_missing_compiled_camera_is_not_silently_accepted(native, monkeypatch):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    _, r, path, _ = native
    create = r.create_bot

    def drop_camera(scene, cfg, context):
        bot = create(scene, cfg, context)
        for handle in list(bot.get_sensor_handles()):
            context.destroy_sensor(handle)
        return bot

    with monkeypatch.context() as patch:
        patch.setattr(r, "create_bot", drop_camera)
        with pytest.raises(RuntimeError, match="inventory"):
            create_backend("superdex", SceneCfg(str(path)), 1, 0.002)
    b = create_backend("superdex", SceneCfg(str(path)), 1, 0.002)
    b.close()


@pytest.mark.parametrize("kind", ["BASIC_JSC_PD", "BASIC_OSC_PD", "MOCHI_ARTICULATED_POSE"])
def test_controller_selective_reset_preserves_other_environment(native, kind):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    p, r, path, _ = native
    b = create_backend("superdex", SceneCfg(str(path)), 2, 0.002)
    single = create_backend("superdex", SceneCfg(str(path)), 1, 0.002)
    try:
        for backend in (b, single):
            backend.configure_controller(**controller_config(r, p, kind, False))
        if kind == "BASIC_JSC_PD":
            target = r.ControllerBasicJscPdTarget(target_pose=[0.25])
        elif kind == "BASIC_OSC_PD":
            obs = b._controllers[0][2].get_current_observations_from_mochi()
            pose = obs.world_from_root.inverse() * obs.world_from_ee_link
            pose.rotation = p.Quaternion.from_rotation_vector([0, 0, 0.25])
            target = r.ControllerBasicOscPdTarget(root_from_target_ee=pose)
        else:
            target = r.ControllerMochiArticulatedPoseTarget()
            target.world_from_root = b._links[0][0].get_root_transform()
            target.pose_dofs = [0.25]
        b.step_controller([target, target], 8)
        single.step_controller([target], 8)
        b.reset(np.array([0]))
        b.step_controller([target, target], 8)
        single.step_controller([target], 8)
        for left, right in zip(qualify._native_state(b, 1), qualify._native_state(single)):
            np.testing.assert_array_equal(left, right)
        for backend in (b, single):
            backend.clear_controller()
            backend.step(np.zeros((backend.num_envs, 1)), 8)
        for left, right in zip(qualify._native_state(b, 1), qualify._native_state(single)):
            np.testing.assert_array_equal(left, right)
    finally:
        single.close()
        b.close()


def test_all_continues_after_blocked_and_failed_models(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.superdex_bot").write_text("{}")
    monkeypatch.setattr(qualify, "resolve_assets_root", lambda value: tmp_path)
    visited, reports = [], []

    def run(path, *args):
        visited.append(path)
        if path.startswith("a"):
            raise NotImplementedError("custom component")
        if path.startswith("b"):
            raise ValueError("invalid model")
        return 0

    monkeypatch.setattr(qualify, "qualify_bot", run)
    monkeypatch.setattr(
        qualify, "write_report", lambda out, path, report, digest: reports.append(report["status"])
    )
    assert qualify.main(["--all", "--skip-verification", "--out", str(tmp_path)]) == 1
    assert visited == ["a.superdex_bot", "b.superdex_bot", "c.superdex_bot"]
    assert reports == ["blocked", "failed"]


@pytest.mark.parametrize("mode", ["serial", "batch"])
def test_controller_holds_external_wrench_for_all_substeps(native, mode):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    _, _, path, _ = native
    controlled = create_backend(
        "superdex", SceneCfg(str(path)), 1, 0.002, superdex_execution_mode=mode
    )
    torque = create_backend("superdex", SceneCfg(str(path)), 1, 0.002, superdex_execution_mode=mode)
    try:
        controlled.configure_controller(
            "BASIC_JSC_PD",
            param_args=json.dumps(
                {
                    "Kp": [0.0],
                    "Kd": [0.0],
                    "saturation": [2.0],
                    "deadband": [0.0],
                }
            ),
        )
        target = controlled._r.ControllerBasicJscPdTarget(target_pose=[0.1])
        for backend in (controlled, torque):
            backend.apply_body_force(
                np.array([2]), np.zeros((1, 1, 3)), np.array([[[0.0, 0.0, 0.5]]])
            )
        controlled.step_controller([target], 8)
        torque.step(np.zeros((1, 1)), 8)
        for left, right in zip(qualify._native_state(controlled), qualify._native_state(torque)):
            np.testing.assert_array_equal(left, right)
        assert not controlled._pending_wrench.any()
    finally:
        torque.close()
        controlled.close()
