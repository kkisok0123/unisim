"""Built-in component audits and optional native camera/controller qualification."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.superdex.components import audit_components

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import superdex_compare as compare  # noqa: E402


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_viewer_propagates_selected_asset_root(tmp_path, monkeypatch, dual, explicit):
    import superdex_viewer as viewer

    from unisim.backend.superdex.scenes import _reference, _root

    monkeypatch.setattr(viewer, "REPOSITORY_ROOT", tmp_path)
    root = tmp_path / ("custom-assets" if explicit else "assets/superdex")
    model = root / "benchmarks/cart_pole/cart_pole.mochi_scene"
    model.parent.mkdir(parents=True)
    model.write_text("{}")
    mesh = model.with_name("mesh_000.mochi.h5")
    mesh.touch()
    if explicit:
        monkeypatch.setenv("SUPERDEX_ASSETS_PATH", str(tmp_path / "other-assets"))
    else:
        monkeypatch.delenv("SUPERDEX_ASSETS_PATH", raising=False)
    previous = os.environ.get("SUPERDEX_ASSETS_PATH")
    calls = []

    def run(path, args, assets_root):
        calls.append(path)
        assert assets_root == root
        assert _root(path) == compare._root_of(path) == root
        assert _reference(path, "benchmarks/cart_pole/mesh_000.mochi.h5", _root(path)) == mesh
        return 0

    monkeypatch.setattr(viewer, "run_compare" if dual else "run_single", run)
    argv = ["benchmarks/cart_pole/cart_pole.mochi_scene",
            "--controlled-joints", "Cart", "--effort-limit", "3.0"]
    if explicit:
        argv += ["--assets", str(root)]
    if dual:
        argv += ["--compare"]
    assert viewer.main(argv) == 0
    assert calls == [model]
    assert os.environ.get("SUPERDEX_ASSETS_PATH") == previous


def test_viewer_external_model_keeps_local_dependency_root(tmp_path, monkeypatch):
    import superdex_viewer as viewer

    from unisim.backend.superdex.scenes import _root

    model = tmp_path / "external.mochi_scene"
    model.write_text("{}")
    monkeypatch.delenv("SUPERDEX_ASSETS_PATH", raising=False)

    def run(path, args, assets_root):
        assert _root(path) == compare._root_of(path) == tmp_path
        return 0

    monkeypatch.setattr(viewer, "run_single", run)
    assert viewer.main([str(model)]) == 0
    assert "SUPERDEX_ASSETS_PATH" not in os.environ


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


def test_component_free_bots_and_explicit_cli_selection():
    assert audit_components([SimpleNamespace(actuators=[], sensors=[])]) == ()
    with pytest.raises(SystemExit) as exc:
        compare.main([])
    assert exc.value.code == 2


def test_static_classification_rejects_custom_components_without_sdk(tmp_path):
    """The comparison tool classifies custom-component bots SDK-free."""
    bot = tmp_path / "custom.superdex_bot"
    bot.write_text(
        json.dumps(
            {
                "name": "custom",
                "joints": [
                    {"name": "root", "type": "Hard"},
                    {"name": "j", "type": "Revolute", "axis": [0, 0, 1]},
                ],
                "links": [
                    {"name": "base", "mass": 1},
                    {
                        "name": "tip",
                        "parentLink": 0,
                        "mass": 1,
                        "actuators": [
                            {
                                "name": "servo",
                                "type": "MY_VELOCITY_SERVO_ACTUATOR",
                                "params": {"k_p": 1},
                            }
                        ],
                    },
                ],
            }
        )
    )
    ref = compare.ModelRef(bot, "bot", bot.name)
    limitation = compare.classify_static(ref)
    assert limitation is not None
    record = limitation.record()
    assert record["blocker_class"] == compare.ADAPTER_RESTRICTION
    assert "MY_VELOCITY_SERVO_ACTUATOR" in record["diagnostic"]
    assert record["checked"] == compare.CHECKED_STATIC

    deformable = tmp_path / "soft.mochi_prefab"
    deformable.write_text(
        json.dumps(
            {
                "actors": {"soft": [{"name": "pillow", "shape": "./pillow.mochi.h5"}]},
            }
        )
    )
    ref = compare.ModelRef(deformable, "prefab", deformable.name)
    limitation = compare.classify_static(ref)
    assert limitation is not None
    assert "deformable" in limitation.record()["unsupported_feature"]

    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    ref = compare.ModelRef(urdf, "urdf", urdf.name)
    limitation = compare.classify_static(ref)
    assert limitation is not None
    assert limitation.record()["blocker_class"] == compare.ADAPTER_RESTRICTION
    assert "URDF" in limitation.record()["enabling_capability"]


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
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 2 3 4\nf 1 4 3\nf 1 3 2\nf 1 2 4\n"
    )
    data["links"][0]["shape"] = "./shape.obj"
    data["links"][1]["shape"] = "./shape.obj"
    path.write_text(json.dumps(data))
    (tmp_path / ".superdex_root").write_text("")
    return p, r, path, floating


def _backend(path, *, floating=False, num_envs=1, mode="serial"):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    kwargs = {"superdex_execution_mode": mode, "superdex_num_workers": 0}
    if not floating:
        kwargs["superdex_effort_limits"] = 2.0
    return create_backend("superdex", SceneCfg(str(path)), num_envs, 0.002, **kwargs)


@pytest.mark.parametrize("native", [False, True], indirect=True)
def test_camera_metadata_and_poses_track_state(native):
    p, r, path, floating = native
    b = _backend(path, floating=floating, num_envs=2)
    try:
        assert b.get_camera_names() == ("camera",)
        parameters = b.get_camera_parameters("camera")
        assert parameters["image_width"] == 320
        assert parameters["image_height"] == 240
        first = b.get_camera_poses("camera")
        assert first.shape == (2, 7)
        assert np.isfinite(first).all()
        with pytest.raises(ValueError, match="camera data"):
            b.get_sensor_data("camera")
        if not floating:
            b.step(np.full((2, 1), 1.0), 10)
            moved = b.get_camera_poses("camera")
            assert not np.array_equal(moved, first)
        # Parameters are detached copies.
        parameters["image_width"] = 640
        assert b.get_camera_parameters("camera")["image_width"] == 320
    finally:
        b.close()


@pytest.mark.parametrize("native", [False], indirect=True)
def test_controller_lifecycle_and_interlocks(native):
    p, r, path, floating = native
    from unisim import JointTarget

    b = _backend(path)
    try:
        # step() before configure_controller fails closed.
        with pytest.raises(RuntimeError):
            b.step_controller([JointTarget(positions=[0.0])])
        b.configure_controller(
            "BASIC_JSC_PD",
            param_args=json.dumps(
                {"Kp": [10.0], "Kd": [1.0], "saturation": [2.0], "deadband": [0.0]}
            ),
        )
        # Double configuration requires clear_controller() first.
        with pytest.raises(RuntimeError, match="clear_controller"):
            b.configure_controller("BASIC_JSC_PD", param_args="{}")
        with pytest.raises(RuntimeError, match="pre-step"):
            b.set_pre_step_control(lambda backend, ctrl: ctrl)
        b.step_controller([JointTarget(positions=[0.2])], 4)
        # NaN targets are rejected without mutating state.
        before = b.get_state()["qpos"].copy()
        with pytest.raises(ValueError, match="finite"):
            b.step_controller([JointTarget(positions=[float("nan")])], 1)
        np.testing.assert_array_equal(b.get_state()["qpos"], before)
        b.clear_controller()
        # step() works again after clearing.
        b.step(np.zeros((1, 1)), 2)
    finally:
        b.close()


@pytest.mark.parametrize("native", [False], indirect=True)
def test_missing_compiled_camera_is_not_silently_accepted(native, monkeypatch):
    p, r, path, floating = native
    from unisim import create_backend
    from unisim.scene import SceneCfg

    real_create = r.create_bot

    def create_without_sensors(scene, cfg, context):
        bot = real_create(scene, cfg, context)
        for handle in list(bot.get_sensor_handles()):
            context.destroy_sensor(handle)
        return bot

    monkeypatch.setattr(r, "create_bot", create_without_sensors)
    with pytest.raises(RuntimeError, match="inventory"):
        create_backend(
            "superdex",
            SceneCfg(str(path)),
            1,
            0.002,
            superdex_execution_mode="serial",
            superdex_num_workers=0,
            superdex_effort_limits=2.0,
        )


def test_unsupported_direct_model_format_is_explicit(tmp_path, capsys):
    """The comparison tool names URDF inputs explicitly and exits nonzero."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    result = compare.main([str(urdf)])
    assert result == 2
    err = capsys.readouterr().err
    assert "unsupported direct-model format" in err
    assert "URDF" in err


def test_discovery_covers_native_formats_and_ignores_payloads(tmp_path):
    root = tmp_path / "bundle"
    (root / "bots").mkdir(parents=True)
    (root / "prefabs").mkdir()
    (root / ".superdex_root").write_text("")
    (root / "bots" / "a.superdex_bot").write_text("{}")
    (root / "bots" / "b.superdex_bot_archive").write_bytes(b"PK")
    (root / "prefabs" / "c.mochi_prefab").write_text("{}")
    (root / "prefabs" / "mesh.mochi.h5").write_bytes(b"\0")
    (root / "package.xml").write_text("<package/>")
    refs = compare.discover_models([root])
    labels = [ref.label.split(":", 1)[1] for ref in refs]
    assert labels == [
        "bots/a.superdex_bot",
        "bots/b.superdex_bot_archive",
        "prefabs/c.mochi_prefab",
    ]
    assert {ref.kind for ref in refs} == {"bot", "archive", "prefab"}


def test_worker_subprocess_failure_is_isolated_and_reported(tmp_path):
    """A crashing worker is reported as failed/crashed, never silent."""
    task = compare.WorkerTask(
        label="missing", path=str(tmp_path / "missing.superdex_bot"), kind="bot"
    )
    result = compare.run_model_subprocess(task, timeout=120)
    assert result["status"] in {"failed", "crashed"}
    assert "missing" in result["model"]


def test_compare_tool_reports_limited_to_out_directory(tmp_path):
    """JSON reports are written only with --out, into the given directory."""
    fr3 = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "superdex"
        / "bots/arms/fr3_v2/fr3_v2.superdex_bot"
    )
    if not fr3.is_file():
        pytest.skip("local asset bundle not present")
    task = compare.WorkerTask(label="fr3_v2", path=str(fr3), kind="bot", steps=1000)
    results = [compare.run_model_subprocess(task, timeout=900)]
    compare.write_reports(tmp_path, results, [])
    data = json.loads((tmp_path / "comparison.json").read_text())
    assert data["counts"]["passed"] == 1
    assert data["provenance"]["implementation_sha256"]


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


def test_pose_controller_link_transform_targets(native):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    p, r, path, _ = native
    b = create_backend("superdex", SceneCfg(str(path)), 1, 0.002)
    reference = compare.BotReference(path)
    try:
        config = controller_config(r, p, "MOCHI_ARTICULATED_POSE", False)
        b.configure_controller(**config)
        reference.configure(config)
        compare.NativeComparison(b, reference.scene, [reference.actor]).seed_reference()
        poses = [link.get_root_transform() for link in b._links[0]]
        t = r.ControllerMochiArticulatedPoseTarget()
        t.world_from_root = poses[0]
        t.local_to_parent_transforms = [p.TransformRT(), poses[0].inverse() * poses[1]]
        b.step_controller([t], 4)
        for _ in range(4):
            reference.step_controller(t, np.array([2.0]))
        for left, right in zip(_native_state(b), reference.qvel()):
            np.testing.assert_array_equal(left, right)
        assert np.isfinite(b.get_state()["qpos"]).all()
        b.reset()
        b.step_controller([t])
    finally:
        reference.close()
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
        for left, right in zip(_native_state(b, 1), _native_state(single)):
            np.testing.assert_array_equal(left, right)
        for backend in (b, single):
            backend.clear_controller()
            backend.step(np.zeros((backend.num_envs, 1)), 8)
        for left, right in zip(_native_state(b, 1), _native_state(single)):
            np.testing.assert_array_equal(left, right)
    finally:
        single.close()
        b.close()


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
        for left, right in zip(_native_state(controlled), _native_state(torque)):
            np.testing.assert_array_equal(left, right)
        assert not controlled._pending_wrench.any()
    finally:
        torque.close()
        controlled.close()


def _native_state(backend, env=0):
    return compare.native_state(compare.native_actors(backend, env)[0])


@pytest.mark.parametrize("native", [False, True], indirect=True)
@pytest.mark.parametrize("kind", ["BASIC_JSC_PD", "BASIC_OSC_PD", "MOCHI_ARTICULATED_POSE"])
def test_controller_comparison_all_supported_targets(native, kind):
    from unisim.backend.superdex.runtime import acquire_runtime, release_runtime

    p, r, path, floating = native
    config = controller_config(r, p, kind, floating)
    acquire_runtime(p)
    try:
        if floating and kind == "BASIC_OSC_PD":
            reference = compare.BotReference(path)
            backend = _backend(path, floating=True)
            try:
                with pytest.raises(Exception, match="bot-space vs actor-space"):
                    reference.configure(config)
                with pytest.raises(NotImplementedError, match="indexing mismatch"):
                    backend.configure_controller(**config)
            finally:
                reference.close()
                backend.close()
            return
        identity = {"translation": [0, 0, 0.5 if floating else 0], "rotation_xyzw": [0, 0, 0, 1]}
        target = {"type_name": kind}
        if kind == "BASIC_JSC_PD":
            target["target_pose"] = ([0, 0, 0.5, 0, 0, 0] if floating else []) + [0.2]
        elif kind == "BASIC_OSC_PD":
            target["root_from_target_ee"] = dict(identity, translation=[0, 0, 0.3])
        else:
            target.update(world_from_root=identity, pose_dofs=[0.2])
        run = compare.CheckRun("fixture", "bot")
        compare.check_controller(compare.ModelRef(path, "bot", "fixture"), config, target, run)
        assert not run.failures, run.checks
        assert run.checks["controller_trajectory"]["physics_steps"] == 1040
        assert run.checks["cameras"]["camera_count"] == 1
    finally:
        release_runtime(p)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--controller-config", "x"],
        ["--absolute-target", "x"],
        ["--all", "--steps", "0"],
        ["--all", "--timeout", "0"],
    ],
)
def test_invalid_comparison_options_fail_before_execution(arguments):
    with pytest.raises(SystemExit) as exc:
        compare.main(arguments)
    assert exc.value.code == 2


def test_multi_articulation_comparison_detects_missing_reference_controls(tmp_path, monkeypatch):
    from unisim.backend.superdex.runtime import acquire_runtime, release_runtime

    p = pytest.importorskip("superdex.physics")
    path = compare.synthetic_fixtures(tmp_path)["multiple"]
    monkeypatch.setattr(compare, "apply_native_control", lambda *args: None)
    acquire_runtime(p)
    try:
        run = compare.CheckRun("missing_controls", "scene")
        compare.check_scene(compare.ModelRef(path, "scene", "missing_controls"), run, steps=1000)
        assert "trajectory_equivalence" in run.failures
    finally:
        release_runtime(p)


def test_viewer_compare_reports_child_configuration_failure(native, tmp_path):
    import superdex_viewer as viewer

    path = native[2]
    args = SimpleNamespace(
        model=str(path),
        controller_config=tmp_path / "missing.json",
        absolute_target=tmp_path / "missing_target.json",
        controlled_joints=None,
        effort_limit=2.0,
        fragments=[],
        no_gravity=False,
        frames=6,
        out=None,
    )
    assert viewer.run_compare(path, args, tmp_path) == 1


def test_relocated_controller_example_paths():
    root = Path(__file__).resolve().parents[1]
    if not (
        root / "assets/superdex/bots/arms/fr3_v2/control/fr3_v2_pose.superdex_controller"
    ).is_file():
        pytest.skip("local asset bundle not present")
    for name in ("fr3-jsc", "fr3-osc", "fr3-pose", "fr3-authored-pose"):
        config = compare.load_controller_config(root / "docs/superdex-configs" / (name + ".json"))
        assert config["type_name"]
