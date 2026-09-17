#!/usr/bin/env python3
"""Qualify local SuperDex bots, cameras and controllers against direct SDK scenes.

Use --bots PATH[,PATH] or --all. Optional --scene adds rigid prefab fragments;
--effort-limit supplies missing authored torque limits. --controller-config
selects one built-in controller using SDK param_args/init_args strings.
--absolute-target holds a controller-specific absolute JSON command, and
--visualize opens synchronized direct-SDK and UniSim-adapter viewer windows.
No assets or optional SDKs are downloaded by this command.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import queue
import signal
import sys
import time
from pathlib import Path
from threading import BrokenBarrierError
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from superdex_bot_qualify import (  # noqa: E402
    INVENTORY_REPORT,
    _git,
    _load_prefab,
    authored_joint_metadata,
    resolve_assets_root,
)

from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402

SIM_DT = 0.002
DECIMATION = 8
SWEEP_CTRL_STEPS = 130
TOL_TRAJECTORY = 1e-7
TOL_RESET = 1e-6
VIEWER_SIZE = (640, 640)
CAMERA_FIELDS = (
    "name",
    "image_width",
    "image_height",
    "fov_vertical_deg",
    "near_clip",
    "far_clip",
    "forward_axis",
    "up_axis_local",
    "offset_local",
    "look_at",
    "look_distance",
)


def audit(cfg) -> None:
    """Independent reference audit: do not import adapter component implementations."""
    for link in cfg.links:
        for item in link.actuators:
            raise NotImplementedError(
                f"unsupported actuator component {item.name!r}: {item.type!r}"
            )
        for item in link.sensors:
            if item.type != "SENSOR_CAMERA":
                raise NotImplementedError(
                    f"unsupported sensor component {item.name!r}: {item.type!r}"
                )


def _native_state(backend, env=0):
    actor = backend._actors[env]
    q = np.empty(backend.model.robot_nv, dtype=backend._dtype)
    v = np.empty_like(q)
    actor.get_articulated_pose(q)
    actor.get_articulated_joint_velocities(v)
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise RuntimeError("non-finite adapter state during qualification")
    return q, v


def _pose(actor):
    t = actor.get_root_transform()
    return np.r_[t.translation, t.rotation]


def _finite_vector(value, size, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape ({size},)")
    return array


def _absolute_transform(p, value, label):
    if not isinstance(value, dict) or set(value) != {"translation", "rotation_xyzw"}:
        raise ValueError(f"{label} requires translation and rotation_xyzw")
    translation = _finite_vector(value["translation"], 3, f"{label} translation")
    rotation = _finite_vector(value["rotation_xyzw"], 4, f"{label} rotation_xyzw")
    if not np.isclose(np.linalg.norm(rotation), 1, atol=1e-5):
        raise ValueError(f"{label} rotation_xyzw must be normalized")
    return p.TransformRT(translation=translation, rotation=rotation)


def absolute_target(reference, spec):
    """Build one SDK target from absolute, controller-specific JSON values."""
    if not isinstance(spec, dict) or spec.get("type_name") != reference.kind:
        raise ValueError("absolute target type_name must match the configured controller")
    kind, r = reference.kind, reference.r
    if kind == "BASIC_JSC_PD":
        if set(spec) != {"type_name", "target_pose"}:
            raise ValueError("JSC absolute target requires only target_pose")
        pose = _finite_vector(spec["target_pose"], reference.nv, "JSC target_pose")
        return r.ControllerBasicJscPdTarget(target_pose=pose.astype(reference.dtype))
    if kind == "BASIC_OSC_PD":
        if set(spec) != {"type_name", "root_from_target_ee"}:
            raise ValueError("OSC absolute target requires only root_from_target_ee")
        pose = _absolute_transform(
            reference.p, spec["root_from_target_ee"], "OSC root_from_target_ee"
        )
        return r.ControllerBasicOscPdTarget(root_from_target_ee=pose)
    if kind == "MOCHI_ARTICULATED_POSE":
        if set(spec) != {"type_name", "world_from_root", "pose_dofs"}:
            raise ValueError(
                "articulated-pose absolute target requires world_from_root and pose_dofs"
            )
        target = r.ControllerMochiArticulatedPoseTarget()
        target.world_from_root = _absolute_transform(
            reference.p, spec["world_from_root"], "pose world_from_root"
        )
        target.pose_dofs = _finite_vector(spec["pose_dofs"], reference.n, "pose pose_dofs").astype(
            reference.dtype
        )
        return target
    raise NotImplementedError(f"unsupported absolute target controller {kind!r}")


def _open_viewer(scene, title):
    from superdex.physics.viewer import VIEWER_AVAILABLE, Viewer, ViewerCfg

    if not VIEWER_AVAILABLE:
        raise RuntimeError("live comparison requires Polyscope >= 2.5.0")
    import polyscope as ps

    ps.set_program_name(title)
    viewer = Viewer(ViewerCfg(offscreen=False, size=VIEWER_SIZE))
    viewer.set_scene(scene)
    direction = np.asarray([1.0, -1.0, 0.6])
    viewer.frame_scene(look_dir=direction / np.linalg.norm(direction))
    return viewer


class Reference:
    """Independent SDK loop; no adapter registration, arithmetic or stepping helpers."""

    def __init__(self, path, fragments, root):
        import superdex.physics as p
        import superdex.robotics as r

        self.p, self.r = p, r
        self.cfg = _load_prefab(path)
        audit(self.cfg)
        self.scene = p.create_scene("qualification_reference")
        self.scene.set_gravity([0, 0, -9.81])
        self.bot = None
        self.controller = None
        self.kind = None
        try:
            self.context = r.create_context()
            self.bot = r.create_bot(self.scene, self.cfg, self.context)
            self.actor = self.bot.get_articulated_actor()
            self.links = [self.scene.get_actor(h) for h in self.actor.get_nested_link_actors()]
            self.cameras = {
                s.get_name(): s
                for s in (self.bot.get_sensor(h) for h in self.bot.get_sensor_handles())
            }
            expected = {s.name for link in self.cfg.links for s in link.sensors}
            if set(self.cameras) != expected or list(self.bot.get_actuator_handles()):
                raise RuntimeError("reference SDK skipped authored components")
            self.rigids = []
            for path in fragments:
                prefab = p.prefab.load_from_file(str(path), str(root))
                self.rigids.extend(p.prefab.add_to_scene(prefab, self.scene).actors)
            self.snapshot = self.scene.capture_state()
            self.dtype = np.float64 if p.uses_double_precision() else np.float32
            self.nv = self.actor.get_num_dofs()
            self.n = len(authored_joint_metadata(self.cfg)[0])
            self.offset = self.nv - self.n
        except BaseException:
            self.close()
            raise

    def state(self):
        q = np.empty(self.nv, dtype=self.dtype)
        v = np.empty_like(q)
        self.actor.get_articulated_pose(q)
        self.actor.get_articulated_joint_velocities(v)
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError("non-finite reference state during qualification")
        return q, v

    def seed(self, q, v):
        if self.controller is not None and self.kind == "MOCHI_ARTICULATED_POSE":
            self.actor.remove_articulated_pose_controller()
        self.scene.restore_state(self.snapshot, release_immediately=False)
        self.actor.set_articulated_pose_from_joints(q)
        self.actor.set_articulated_joint_velocities(v)
        self.actor.set_external_forces_on_dofs(np.arange(self.nv, dtype=np.int32), np.zeros_like(q))
        self.scene.step(0)
        if self.controller is not None:
            self.controller.reset()
            if self.kind == "MOCHI_ARTICULATED_POSE":
                self.controller.initialize(False)

    def configure(self, config):
        self.kind = config["type_name"]
        self.controller = self.bot.create_controller(self.kind)
        self.controller.configure_from_scene_entry(config["param_args"], config["init_args"])

    def step(self, command, limits):
        if self.controller is None:
            effort = np.r_[np.zeros(self.offset), command].astype(self.dtype)
        elif self.kind == "MOCHI_ARTICULATED_POSE":
            self.controller.compute_output(self.r.ControllerMochiArticulatedPoseObsv(), command)
            effort = np.zeros(self.nv, dtype=self.dtype)
        else:
            obsv = self.controller.get_current_observations_from_mochi()
            if self.kind == "BASIC_JSC_PD":
                obsv.dt = SIM_DT
            effort = np.asarray(self.controller.compute_output(obsv, command)).copy()
            if np.any(effort[: self.offset] != 0):
                raise ValueError("reference controller actuates floating root")
        effort[self.offset :] = np.clip(effort[self.offset :], -limits, limits)
        self.actor.set_external_forces_on_dofs(np.arange(self.nv, dtype=np.int32), effort)
        self.scene.step(SIM_DT)

    def close(self):
        if self.bot is not None:
            if (
                self.kind == "MOCHI_ARTICULATED_POSE"
                and self.actor.has_articulated_pose_controller()
            ):
                self.actor.remove_articulated_pose_controller()
            self.r.destroy_bot(self.scene, self.bot)
            self.bot = None
        if hasattr(self, "snapshot"):
            self.scene.release_state(self.snapshot)
        self.p.destroy_scene(self.scene)


def _resolve_effort_limits(cfg, fallback):
    names, _, authored, _ = authored_joint_metadata(cfg)
    if not names:
        raise NotImplementedError("qualification requires scalar joints")
    limits = np.asarray(authored)
    if np.any(limits <= 0) or not np.isfinite(limits).all():
        if fallback is None:
            raise NotImplementedError("authored effort limits are -1/0; supply --effort-limit")
        limits = np.asarray(fallback, dtype=float)
        if limits.size == 1:
            limits = np.repeat(limits, len(names))
    if limits.shape != (len(names),) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("effort limits must be finite positive, one value or one per joint")
    return limits


def _wait_for_live_frame(frame_barrier, stop_event):
    try:
        frame_barrier.wait(timeout=2)
    except BrokenBarrierError:
        return False
    return not stop_event.is_set()


def _run_adapter_viewer(
    path,
    fragments,
    limits,
    config,
    target_spec,
    ready_queue,
    start_event,
    stop_event,
    frame_barrier,
):
    """Run the UniSim adapter in its own process and own one live renderer."""
    from unisim import create_backend
    from unisim.scene import SceneCfg

    role = "UniSim adapter"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend = viewer = None
    try:
        backend = create_backend(
            "superdex",
            SceneCfg(str(path), fragment_files=[Path(p) for p in fragments]),
            1,
            SIM_DT,
            superdex_effort_limits=limits,
            superdex_execution_mode="serial",
            superdex_num_workers=0,
        )
        backend.configure_controller(**config)
        backend.reset()
        target = absolute_target(
            SimpleNamespace(
                kind=config["type_name"],
                r=backend._r,
                p=backend._p,
                nv=backend.model.robot_nv,
                n=backend.num_actuators,
                dtype=backend._dtype,
            ),
            target_spec,
        )
        viewer = _open_viewer(backend._worlds[0], role)
        ready_queue.put((role, "ready", ""))
        start_event.wait()
        while not stop_event.is_set() and _wait_for_live_frame(frame_barrier, stop_event):
            started = time.monotonic()
            backend.step_controller([target], DECIMATION)
            viewer.render()
            if viewer.user_requested_close():
                break
            stop_event.wait(max(0.0, SIM_DT * DECIMATION - (time.monotonic() - started)))
    except BaseException as exc:
        ready_queue.put((role, "error", f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        stop_event.set()
        if viewer is not None:
            viewer.close()
        if backend is not None:
            backend.close()


def _run_native_viewer(
    path,
    fragments,
    root,
    limits,
    config,
    target_spec,
    ready_queue,
    start_event,
    stop_event,
    frame_barrier,
):
    """Run the direct SuperDex SDK path in its own process and own one live renderer."""
    import superdex.physics as p

    role = "Native SuperDex SDK"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    initialized = False
    reference = viewer = None
    try:
        p.initialize(num_worker_threads=0)
        initialized = True
        reference = Reference(Path(path), [Path(item) for item in fragments], Path(root))
        limits = np.asarray(limits, dtype=reference.dtype)
        reference.configure(config)
        reference.seed(*reference.state())
        target = absolute_target(reference, target_spec)
        viewer = _open_viewer(reference.scene, role)
        ready_queue.put((role, "ready", ""))
        start_event.wait()
        while not stop_event.is_set() and _wait_for_live_frame(frame_barrier, stop_event):
            started = time.monotonic()
            for _ in range(DECIMATION):
                reference.step(target, limits)
            viewer.render()
            if viewer.user_requested_close():
                break
            stop_event.wait(max(0.0, SIM_DT * DECIMATION - (time.monotonic() - started)))
    except BaseException as exc:
        ready_queue.put((role, "error", f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        stop_event.set()
        if viewer is not None:
            viewer.close()
        if reference is not None:
            reference.close()
        if initialized:
            p.shutdown()


def run_live_comparison(relpath, root, fragments, effort_fallback, config, target_spec):
    """Open two synchronized viewers and keep them running until either one closes."""
    import multiprocessing

    if config is None or target_spec is None:
        raise ValueError("live comparison requires a controller config and absolute target")
    path = root / relpath
    cfg = _load_prefab(path)
    audit(cfg)
    limits = _resolve_effort_limits(cfg, effort_fallback).tolist()
    fragment_paths = [str(root / item) for item in fragments or []]
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    start_event = context.Event()
    stop_event = context.Event()
    frame_barrier = context.Barrier(2)
    common = (
        str(path),
        fragment_paths,
        limits,
        config,
        target_spec,
        ready_queue,
        start_event,
        stop_event,
        frame_barrier,
    )
    processes = [
        context.Process(target=_run_adapter_viewer, args=common, name="unisim-adapter-viewer"),
        context.Process(
            target=_run_native_viewer,
            args=(str(path), fragment_paths, str(root), *common[2:]),
            name="native-superdex-viewer",
        ),
    ]
    errors = []
    started_processes = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        ready = set()
        deadline = time.monotonic() + 60
        while len(ready) < len(processes):
            try:
                role, status, detail = ready_queue.get(timeout=0.1)
            except queue.Empty:
                failed = [p for p in processes if p.exitcode is not None]
                if failed:
                    errors.append(f"{failed[0].name} exited during viewer startup")
                    break
                if time.monotonic() >= deadline:
                    errors.append("timed out while opening the two viewer windows")
                    break
                continue
            if status == "error":
                errors.append(f"{role}: {detail}")
                break
            ready.add(role)
        if not errors:
            print("Two live windows are running:")
            print(f"  UniSim adapter (PID {processes[0].pid})")
            print(f"  Native SuperDex SDK (PID {processes[1].pid})")
            print("Close either window, or press Ctrl+C here, to stop both.")
            start_event.set()
            while all(process.is_alive() for process in processes):
                try:
                    role, status, detail = ready_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if status == "error":
                    errors.append(f"{role}: {detail}")
                    break
    except KeyboardInterrupt:
        print("Stopping both live viewers...")
    finally:
        stop_event.set()
        start_event.set()
        frame_barrier.abort()
        for process in started_processes:
            process.join(timeout=5)
        for process in started_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
    failed = [process for process in started_processes if process.exitcode not in (0, None)]
    if failed and not errors:
        errors.append(
            ", ".join(f"{process.name} exited with code {process.exitcode}" for process in failed)
        )
    if errors:
        for error in errors:
            print(f"live comparison failed: {error}", file=sys.stderr)
        return 1
    return 0


def _target(reference, q0, amplitudes, k, anchor):
    wave = np.sin(2 * np.pi * k * SIM_DT * DECIMATION / 2 + np.linspace(0, 4, reference.n))
    joints = (q0[reference.offset :] + amplitudes * wave).astype(reference.dtype)
    r = reference.r
    if reference.kind == "BASIC_JSC_PD":
        return r.ControllerBasicJscPdTarget(target_pose=np.r_[q0[: reference.offset], joints])
    if reference.kind == "BASIC_OSC_PD":
        pose = reference.p.TransformRT(
            translation=np.asarray(anchor.translation).copy(), rotation=anchor.rotation
        )
        pose.translation = np.asarray(pose.translation) + [0.01 * np.sin(k * 0.03), 0, 0]
        return r.ControllerBasicOscPdTarget(root_from_target_ee=pose)
    target = r.ControllerMochiArticulatedPoseTarget()
    target.world_from_root = anchor
    target.pose_dofs = joints
    return target


def qualify_bot(
    relpath,
    root,
    out,
    tree_digest=None,
    fragments=None,
    effort_fallback=None,
    controller_config=None,
    absolute_target_spec=None,
):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    if absolute_target_spec is not None and controller_config is None:
        raise ValueError("absolute targets require a configured controller")
    path = root / relpath
    cfg = _load_prefab(path)
    audit(cfg)
    names, ranges, _, armature = authored_joint_metadata(cfg)
    limits = _resolve_effort_limits(cfg, effort_fallback)
    fragment_paths = [root / f for f in fragments or []]
    report = {
        "bot": relpath,
        "scene_fragments": fragments or [],
        "checks": {},
        "control_mode": controller_config["type_name"] if controller_config else "torque",
        "controller_config": controller_config,
        "absolute_target": absolute_target_spec,
        "sim_dt": SIM_DT,
        "decimation": DECIMATION,
        "effort_limits": limits.tolist(),
    }

    def record(name, passed, **evidence):
        report["checks"][name] = {"passed": bool(passed), **evidence}

    def make(count, mode="batch"):
        b = create_backend(
            "superdex",
            SceneCfg(str(path), fragment_files=fragment_paths),
            count,
            SIM_DT,
            superdex_effort_limits=limits.tolist(),
            superdex_execution_mode=mode,
            superdex_num_workers=0,
        )
        try:
            if controller_config:
                b.configure_controller(**controller_config)
            return b
        except BaseException:
            b.close()
            raise

    backend = serial = reference = None
    try:
        backend = make(2)
        serial = make(1, "serial")
        reference = Reference(path, fragment_paths, root)
        if controller_config:
            reference.configure(controller_config)
        backend.reset()
        serial.reset()
        reference.seed(*_native_state(backend))
        q0, _ = reference.state()
        report["root"] = "floating" if reference.offset else "fixed"
        record(
            "structure",
            tuple(backend.model.joint_names) == names
            and tuple(backend.model.body_names[1 : len(cfg.links) + 1])
            == tuple(link.name for link in cfg.links)
            and len(backend._rigids[0]) == len(reference.rigids)
            and all(
                a.get_name() == b.get_name()
                and a.is_static() == b.is_static()
                and (a.is_static() or np.isclose(a.get_mass(), b.get_mass()))
                for a, b in zip(backend._rigids[0], reference.rigids)
            ),
        )
        record(
            "components",
            set(backend.get_camera_names()) == set(reference.cameras),
            cameras=len(reference.cameras),
            custom_actuators=0,
        )
        reach = np.minimum(
            q0[reference.offset :] - ranges[:, 0], ranges[:, 1] - q0[reference.offset :]
        )
        amplitudes = np.minimum(0.25, 0.3 * np.maximum(0, reach))
        kp = np.full(len(names), 60.0)
        kd = 2 * np.sqrt(kp * np.maximum(armature, 1e-3))
        anchor = reference.links[0].get_root_transform()
        if reference.kind == "BASIC_OSC_PD":
            obsv = reference.controller.get_current_observations_from_mochi()
            anchor = obsv.world_from_root.inverse() * obsv.world_from_ee_link
        requested_target = (
            absolute_target(reference, absolute_target_spec) if absolute_target_spec else None
        )
        deviation = 0.0
        rigid_deviation = 0.0
        isolation = 0.0
        for k in range(SWEEP_CTRL_STEPS):
            target = (
                requested_target
                if requested_target is not None
                else _target(reference, q0, amplitudes, k, anchor)
                if controller_config
                else None
            )
            hold = _target(reference, q0, amplitudes * 0, 0, anchor) if controller_config else None
            for _ in range(DECIMATION):
                if controller_config:
                    backend.step_controller([target, hold])
                    serial.step_controller([target])
                    reference.step(target, limits)
                else:
                    goal = q0[reference.offset :] + amplitudes * np.sin(k * 0.03)
                    q, v = reference.state()
                    command = np.clip(
                        kp * (goal - q[reference.offset :]) - kd * v[reference.offset :],
                        -limits,
                        limits,
                    ).astype(reference.dtype)
                    backend.step(np.stack([command, np.zeros_like(command)]))
                    serial.step(command[None])
                    reference.step(command, limits)
                qr, vr = reference.state()
                for b in (backend, serial):
                    qb, vb = _native_state(b)
                    if not all(np.isfinite(x).all() for x in (qb, vb, qr, vr)):
                        raise RuntimeError("non-finite state during trajectory qualification")
                    deviation = max(
                        deviation, float(np.max(np.abs(qb - qr))), float(np.max(np.abs(vb - vr)))
                    )
                    for a, r in zip(b._rigids[0], reference.rigids):
                        if not np.isfinite(_pose(a)).all() or not np.isfinite(_pose(r)).all():
                            raise RuntimeError("non-finite rigid state during qualification")
                        rigid_deviation = max(
                            rigid_deviation, float(np.max(np.abs(_pose(a) - _pose(r))))
                        )
                isolation = max(
                    isolation,
                    float(
                        np.max(
                            np.abs(backend.get_state()["qpos"][0] - serial.get_state()["qpos"][0])
                        )
                    ),
                )
        record(
            "trajectory_equivalence",
            deviation <= TOL_TRAJECTORY and rigid_deviation <= TOL_RESET,
            physics_steps=SWEEP_CTRL_STEPS * DECIMATION,
            native_state_max_dev=deviation,
            rigid_pose_max_dev=rigid_deviation,
        )
        record("isolation", isolation <= TOL_RESET, env0_vs_single_max_dev=isolation)
        camera_dev = 0.0
        params_ok = True
        for name, sensor in reference.cameras.items():
            params = sensor.get_params()
            actual = backend.get_camera_parameters(name)
            for key in CAMERA_FIELDS:
                value = getattr(params, key)
                params_ok &= (
                    actual[key] == value
                    if isinstance(value, str)
                    else np.array_equal(actual[key], np.asarray(value))
                )
            t = sensor.get_world_transform()
            expected = np.r_[t.translation, np.asarray(t.rotation)[[3, 0, 1, 2]]]
            camera_dev = max(
                camera_dev, float(np.max(np.abs(backend.get_camera_poses(name)[0] - expected)))
            )
        if reference.cameras:
            record("cameras", params_ok and camera_dev <= TOL_RESET, pose_max_dev=camera_dev)
        else:
            report["checks"]["cameras"] = {"skipped": "no authored cameras"}
        before = backend.get_state()
        backend.reset(np.array([0]))
        after = backend.get_state()
        reset_ok = np.array_equal(before["qpos"][1], after["qpos"][1]) and np.array_equal(
            before["qvel"][1], after["qvel"][1]
        )
        reset_ok &= np.allclose(after["qpos"][0], backend.get_default_qpos(), atol=TOL_RESET)
        backend.set_state(np.array([0]), before["qpos"][:1], before["qvel"][:1])
        reset_ok &= np.allclose(backend.get_state()["qpos"][0], before["qpos"][0], atol=TOL_RESET)
        reset_ok &= np.allclose(backend.get_state()["qvel"][0], before["qvel"][0], atol=TOL_RESET)
        backend.reset()
        reset_ok &= np.allclose(
            backend.get_state()["qpos"], backend.get_default_qpos(), atol=TOL_RESET
        )
        record("reset", reset_ok)
        if controller_config:
            record("controller", deviation <= TOL_TRAJECTORY, type_name=reference.kind)
            backend.clear_controller()
            serial.clear_controller()
        # Controller-free torque wiring and clipping remain independently qualified.
        if reference.controller is not None:
            if reference.kind == "MOCHI_ARTICULATED_POSE":
                reference.actor.remove_articulated_pose_controller()
            for handle in list(reference.bot.get_controller_handles()):
                reference.context.destroy_controller(handle)
            reference.controller = None
        control_dev = 0.0
        responds = True
        for joint in range(len(names)):
            backend.reset()
            reference.seed(*_native_state(backend))
            command = np.zeros(len(names), dtype=reference.dtype)
            command[joint] = min(0.1, limits[joint] * 0.5)
            for _ in range(5):
                backend.step(np.tile(command, (2, 1)))
                reference.step(command, limits)
            qb, vb = _native_state(backend)
            qr, vr = reference.state()
            control_dev = max(
                control_dev, float(np.max(np.abs(qb - qr))), float(np.max(np.abs(vb - vr)))
            )
            responds &= abs(vb[reference.offset + joint]) > 0
        record(
            "control",
            responds and control_dev <= TOL_TRAJECTORY,
            native_state_max_dev=control_dev,
            per_joint_response=bool(responds),
        )
        backend.reset()
        serial.reset()
        reference.seed(*_native_state(backend))
        for _ in range(8):
            backend.step(np.tile(limits * 10, (2, 1)))
            serial.step(limits[None])
            reference.step(limits, limits)
        qb, vb = _native_state(backend)
        qs, vs = _native_state(serial)
        qr, vr = reference.state()
        clip_dev = max(float(np.max(np.abs(qb - qr))), float(np.max(np.abs(vb - vr))))
        record(
            "effort_clipping",
            np.array_equal(qb, qs) and np.array_equal(vb, vs) and clip_dev <= TOL_TRAJECTORY,
            native_vs_sdk_max_dev=clip_dev,
        )
        for _ in range(3):
            cycle = make(1)
            try:
                if controller_config:
                    cycle.step_controller([_target(reference, q0, amplitudes, 0, anchor)])
                else:
                    cycle.step(np.zeros((1, len(names))))
                cycle.reset()
            finally:
                cycle.close()
        record("lifecycle", True, cycles=3)
    finally:
        if reference is not None:
            reference.close()
        if serial is not None:
            serial.close()
        if backend is not None:
            backend.close()
    report["status"] = (
        "passed" if all(c.get("passed", True) for c in report["checks"].values()) else "failed"
    )
    write_report(out, relpath, report, tree_digest)
    return int(report["status"] != "passed")


def discover_bots(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.superdex_bot"))


def write_report(out, relpath, report, digest=None):
    report["provenance"] = {
        "code_commit": _git("rev-parse", "HEAD"),
        "working_tree_modified": bool(_git("status", "--porcelain")),
        "tree_digest": digest,
        "python": platform.python_version(),
        "sdk_version": importlib.metadata.version("superdex-physics-uni"),
        "tolerances": {"native_state": TOL_TRAJECTORY, "reset": TOL_RESET},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(relpath).stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{relpath}: {report['status']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets")
    parser.add_argument(
        "--out", type=Path, default=REPOSITORY_ROOT / "docs/superdex-component-qualification"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--bots", help="comma-separated bundle-relative .superdex_bot paths")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--scene", help="comma-separated rigid .mochi_prefab paths")
    parser.add_argument("--effort-limit")
    parser.add_argument("--controller-config", type=Path)
    parser.add_argument(
        "--absolute-target",
        type=Path,
        help="controller-specific JSON absolute target; requires --controller-config",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="open synchronized native SDK and UniSim adapter viewer windows",
    )
    parser.add_argument("--skip-verification", action="store_true")
    args = parser.parse_args(argv)
    root = resolve_assets_root(args.assets)
    digest = (
        None
        if args.skip_verification
        else verify_asset_bundle(root, INVENTORY_REPORT)["tree_digest"]
    )
    available = discover_bots(root)
    selected = available if args.all else args.bots.split(",")
    if any(b not in available for b in selected):
        parser.error("unknown bot path")
    limits = [float(v) for v in args.effort_limit.split(",")] if args.effort_limit else None
    if limits is not None and (not np.isfinite(limits).all() or min(limits) <= 0):
        parser.error("effort limits must be finite positive")
    config = None
    if args.controller_config:
        config = json.loads(args.controller_config.read_text())
        if set(config) - {"type_name", "param_args", "init_args"} or "type_name" not in config:
            parser.error("controller config requires type_name and optional param_args/init_args")
        for key in ("param_args", "init_args"):
            value = config.setdefault(key, "")
            if not isinstance(value, str):
                parser.error(f"{key} must be a file path or inline JSON string")
            if value and not value.lstrip().startswith("{"):
                config[key] = str((args.controller_config.parent / value).resolve())
    target_spec = None
    if args.absolute_target:
        if config is None:
            parser.error("--absolute-target requires --controller-config")
        target_spec = json.loads(args.absolute_target.read_text())
        if not isinstance(target_spec, dict):
            parser.error("--absolute-target must contain a JSON object")
        if target_spec.get("type_name") != config["type_name"]:
            parser.error("absolute target type_name must match controller config type_name")
    if args.visualize and target_spec is None:
        parser.error("--visualize requires --absolute-target")
    fragments = args.scene.split(",") if args.scene else None
    if args.visualize:
        if args.all or len(selected) != 1:
            parser.error("--visualize requires exactly one --bots path")
        try:
            return run_live_comparison(selected[0], root, fragments, limits, config, target_spec)
        except Exception as exc:
            print(f"live comparison failed: {exc}", file=sys.stderr)
            return 1
    overall = 0
    for relpath in selected:
        try:
            overall |= qualify_bot(
                relpath,
                root,
                args.out,
                digest,
                fragments,
                limits,
                config,
                target_spec,
            )
        except Exception as exc:
            status = "blocked" if isinstance(exc, NotImplementedError) else "failed"
            write_report(
                args.out,
                relpath,
                {
                    "bot": relpath,
                    "status": status,
                    "blocker" if status == "blocked" else "error": str(exc),
                },
                digest,
            )
            overall = 1
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
