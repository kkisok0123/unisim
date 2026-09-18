#!/usr/bin/env python3
"""Interactive model viewing for every SuperDex adapter input format.

One tool replaces the former per-stage viewers. It accepts the four native
model formats (``.superdex_bot``, ``.superdex_bot_archive``, ``.mochi_scene``,
``.mochi_prefab``) and audited MJCF ``.xml`` inputs, and opens the native
Polyscope viewer through the adapter's public ``run_playback`` API:

* registered bot profiles keep their control/effort settings and demonstrate
  three phases — default-pose hold, bounded sine movement, verified reset;
* unfamiliar supported models are viewed passively (no invented control
  profile): the world simply steps under gravity;
* native scenes/prefabs accept ``--controlled-joints`` and ``--effort-limit``
  for a bounded movement phase, or run passive otherwise;
* ``--compare`` opens two synchronized windows for native inputs — the UniSim
  adapter on one side and a direct SuperDex SDK scene on the other — stepping
  frame-by-frame together; closing either window shuts both down.

Control selection, effort limits, controller configuration and targets,
gravity override, movement and reset verification are preserved. Nothing is
written by default; ``--out`` opts into a JSON evidence report and
``--frames N`` switches to an explicit headless smoke run (no images or
videos are ever saved). Unsupported formats receive an actionable
diagnostic. Asset setup:

    export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
    uv run scripts/superdex_viewer.py bots/arms/fr3_v2/fr3_v2.superdex_bot
    uv run scripts/superdex_viewer.py --list-profiles
    uv run scripts/superdex_viewer.py benchmarks/cart_pole/cart_pole.mochi_scene \\
        --controlled-joints Cart --effort-limit 3.0
    uv run scripts/superdex_viewer.py bots/arms/fr3_v2/fr3_v2.superdex_bot --compare
    uv run scripts/superdex_viewer.py prefabs/sphere/sphere.mochi_prefab  # passive
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from superdex_compare import (  # noqa: E402
    PROFILES,
    _kind_of,
)

SIM_DT = 0.002
CTRL_DT = 1.0 / 60.0
DECIMATION = int(round(CTRL_DT / SIM_DT))

# Demo phase schedule in wall-clock seconds (identical to the historical bot
# viewer); movement amplitude per joint is bounded by the authored ranges.
HOLD_SECONDS = 2.0
MOVE_SECONDS = 6.0
SETTLE_SECONDS = 1.0
MOVE_PERIOD = 2.0
REACHABLE_FRACTION = 0.6
RESET_TOLERANCE = 1e-6

SUPPORTED_SUFFIXES = (
    ".superdex_bot",
    ".superdex_bot_archive",
    ".mochi_scene",
    ".mochi_prefab",
    ".xml",
)


def registered_profile(path):
    from superdex_compare import PROFILES, BotProfile

    for profile in PROFILES.values():
        if path.as_posix().endswith("/" + profile.relpath):
            return profile
    for side in ("left", "right"):
        for kind in ("allegro_v5", "dg5f_short", "dg5f_long", "wuji_hand2_beta1"):
            relative = f"bots/hands/{kind}/{side}/{kind}_{side}.superdex_bot"
            if path.as_posix().endswith("/" + relative):
                return BotProfile(
                    relative,
                    (1.0,) * 20 if kind.startswith("dg5f") else None,
                    kp=0.03,
                    armature_floor=1e-6,
                    sweep_amplitude=0.1,
                )
        relative = f"bots/grippers/openarm_v20/{side}/openarm_v20_{side}_gripper.superdex_bot"
        if path.as_posix().endswith("/" + relative):
            return BotProfile(relative, kp=0.03, armature_floor=1e-6, sweep_amplitude=0.1)
    return None


def backend_options(path, args, profile):
    options = dict(superdex_execution_mode="serial", superdex_num_workers=0)
    if path.suffix in (".mochi_scene", ".mochi_prefab"):
        options["superdex_controlled_joints"] = [
            s.strip() for s in (args.controlled_joints or "").split(",") if s.strip()
        ]
    if args.effort_limit is not None:
        options["superdex_effort_limits"] = args.effort_limit
    elif profile and profile.effort_limits is not None:
        options["superdex_effort_limits"] = profile.effort_limits
    return options


def resolve_model(cli_path: str, assets_root: Path) -> Path:
    """Resolve a CLI model argument against the asset bundle or filesystem."""
    candidate = Path(cli_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    bundled = (assets_root / cli_path).resolve()
    if bundled.is_file():
        return bundled
    raise SystemExit(
        f"error: model not found: {cli_path}\n"
        f"  looked for: {candidate}\n  and:         {bundled}\n"
        "  pass a path relative to the asset root or an absolute file path."
    )


def actionable_format_error(path: Path) -> SystemExit:
    suffix = path.suffix.lower()
    if suffix == ".urdf":
        return SystemExit(
            f"error: {path.name}: URDF is not a supported adapter input.\n"
            "  The adapter loads .superdex_bot, .superdex_bot_archive, .mochi_scene,\n"
            "  .mochi_prefab and audited MJCF .xml; the SDK's own URDF loader also\n"
            "  drops primitive box/cylinder/sphere geometry (see docs/superdex.md)."
        )
    if path.name.endswith(".mochi.h5"):
        return SystemExit(
            f"error: {path.name}: .mochi.h5 files are geometry payloads, not model\n"
            "  inputs; view the referencing .superdex_bot/.mochi_prefab instead."
        )
    return SystemExit(
        f"error: {path.name}: unsupported input format {suffix!r}.\n"
        f"  supported: {', '.join(SUPPORTED_SUFFIXES)}"
    )


# --------------------------------------------------------------------- #
# Shared demo helpers (reused from the comparison tool's vocabulary)
# --------------------------------------------------------------------- #


def _sweep_amplitudes(q0: np.ndarray, ranges: np.ndarray, amplitude: float) -> np.ndarray:
    width = np.minimum(np.abs(ranges[:, 1] - ranges[:, 0]), 2 * np.pi) * REACHABLE_FRACTION
    return np.minimum(amplitude, 0.5 * width) * (width > 0)


def _sweep_target(q0: np.ndarray, amplitudes: np.ndarray, t: float) -> np.ndarray:
    phase = np.linspace(0.0, 3.0 * math.pi / 2.0, len(q0))
    return q0 + amplitudes * np.sin(2.0 * math.pi * t / MOVE_PERIOD + phase)


class Demo:
    """Three-phase movement demonstration against one adapter backend.

    initial-pose → bounded-movement → reset (numerically verified). The
    phase machine is driven from the render loop; each callback advances
    DECIMATION physics substeps at the viewer's 60 Hz pacing.
    """

    def __init__(
        self,
        backend,
        *,
        kp: float,
        kd: np.ndarray,
        effort: np.ndarray,
        amplitudes: np.ndarray,
        floating: bool,
        zero_gravity: bool,
    ):
        self.backend = backend
        self.kp = kp
        self.kd = kd
        self.effort = effort
        self.amplitudes = amplitudes
        self.floating = floating
        # Track actuator-space state through public APIs: scenes may actuate a
        # subset of joints and get_dof_pos returns every coordinate.
        actuator_names = backend.get_actuator_names()
        self.actuator_qpos = np.asarray(
            backend.get_joint_state_qpos_indices(actuator_names), dtype=int
        )
        self.actuator_qvel = np.asarray(
            backend.get_joint_state_qvel_indices(actuator_names), dtype=int
        )
        self.q0 = backend.get_state()["qpos"][0][self.actuator_qpos].copy()
        self.default_state = backend.get_state()["qpos"][0].copy()
        all_ranges = backend.get_joint_range()
        coordinates = backend.get_model_info().coordinate_names
        self.joint_ranges = all_ranges[[coordinates.index(n) for n in actuator_names]]
        self.body_ids = np.arange(1, len(backend.get_model_info().body_names))
        self.support = (
            -backend.get_body_mass()[self.body_ids, None] * backend.get_gravity()[None, :]
        )
        self.phase = "initial-pose"
        self.phase_time = 0.0
        self.movement_peak = 0.0
        self.root_movement_peak = 0.0
        self.reset_error: float | None = None
        self.zero_gravity = zero_gravity
        # The PD always runs: fixed-base bots need it for the sweep (targets
        # are positions, not torques), scenes need it to overcome authored
        # joint friction, and floating/zero-g models additionally use the
        # gravity-compensation branch in step().
        backend.set_pre_step_control(self._position_control)

    def _position_control(self, backend, ctrl):
        state = backend.get_state()
        q = state["qpos"][0][self.actuator_qpos]
        v = state["qvel"][0][self.actuator_qvel]
        # Gains arrive in full-model order (armature per DoF); slice them to
        # the actuator set so the shapes match actuator-space state.
        kd = np.asarray(self.kd)[: len(q)] if len(self.kd) != len(q) else self.kd
        effort = np.asarray(self.effort)[: len(q)] if len(self.effort) != len(q) else self.effort
        torque = np.clip(self.kp * (ctrl[0] - q) - kd * v, -effort, effort)
        return torque[None, :]

    def _enter_phase(self, name: str) -> None:
        if self.phase != name or self.phase_time > 0.0:
            return
        if name == "initial-pose":
            print(f"[{name}] holding authored default pose (inspect geometry)")
        elif name == "bounded-movement":
            if self.floating:
                state = self.backend.get_state()
                state["qvel"][0, :3] = [0.015, 0, 0.005]
                state["qvel"][0, 3:6] = [0, 0.04, 0]
                self.backend.set_state(np.array([0]), state["qpos"], state["qvel"])
            amps = [round(float(a), 3) for a in self.amplitudes]
            print(f"[{name}] sine sweep, amplitudes {amps} rad")
        elif name == "reset":
            self.backend.reset()
            self.reset_error = float(
                np.max(np.abs(self.backend.get_state()["qpos"][0] - self.default_state))
            )
            print(
                f"[{name}] backend.reset() restored default pose "
                f"(max qpos error {self.reset_error:.2e} rad)"
            )

    def step(self, obs):
        self._enter_phase(self.phase)
        target = self.q0.copy()
        if self.phase == "initial-pose":
            if self.phase_time >= HOLD_SECONDS:
                self.phase, self.phase_time = "bounded-movement", -CTRL_DT
        elif self.phase == "bounded-movement":
            target = _sweep_target(self.q0, self.amplitudes, self.phase_time)
            if self.floating:
                target = np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
                self.root_movement_peak = max(
                    self.root_movement_peak,
                    float(
                        np.linalg.norm(
                            self.backend.get_state()["qpos"][0, :3] - self.default_state[:3]
                        )
                    ),
                )
            self.movement_peak = max(
                self.movement_peak,
                float(
                    np.max(
                        np.abs(self.backend.get_state()["qpos"][0][self.actuator_qpos] - self.q0)
                    )
                ),
            )
            if self.phase_time >= MOVE_SECONDS:
                self.phase, self.phase_time = "reset", -CTRL_DT
        elif self.phase == "reset":
            if self.phase_time >= SETTLE_SECONDS:
                settled = float(
                    np.max(
                        np.abs(self.backend.get_state()["qpos"][0][self.actuator_qpos] - self.q0)
                    )
                )
                print(
                    f"[done] reset verified (settle deviation {settled:.3f} rad); "
                    "close the window to exit"
                )
                self.phase, self.phase_time = "done", -CTRL_DT
        if self.floating:
            self.backend.apply_body_force(
                self.body_ids, self.support[None], np.zeros_like(self.support)[None]
            )
        self.backend.step(target[None, :], DECIMATION)
        self.phase_time += CTRL_DT
        return obs

    def evidence(self) -> dict[str, Any]:
        return {
            "movement_peak_rad": self.movement_peak,
            "root_displacement_peak_m": self.root_movement_peak,
            "reset_error_rad": self.reset_error,
        }


class PassiveView:
    """Steps the world under gravity without inventing a control profile."""

    def __init__(self, backend):
        self.backend = backend
        self.default_state = backend.get_state()["qpos"][0].copy()

    def step(self, obs):
        n = self.backend.num_actuators
        self.backend.step(np.zeros((1, n)) if n else np.zeros((1, 0)), DECIMATION)
        return obs

    def evidence(self) -> dict[str, Any]:
        state = self.backend.get_state()["qpos"][0]
        return {
            "displacement_m": float(np.max(np.abs(state - self.default_state)))
            if state.size
            else 0.0,
        }


# --------------------------------------------------------------------- #
# Camera framing
# --------------------------------------------------------------------- #


def compute_camera(backend, direction=(1.0, 1.0, 1.0)):
    from unisim import CameraCfg

    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    body_ids = np.arange(1, len(backend.get_model_info().body_names))
    positions = backend.get_body_pos_w(body_ids)[0]
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    return CameraCfg(
        cam_lookat=tuple((lo + hi) / 2),
        cam_distance=max(0.25, float(np.linalg.norm(hi - lo)) * 2),
        cam_azimuth=float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
        cam_elevation=-float(np.rad2deg(np.arcsin(direction[2]))),
    )


# --------------------------------------------------------------------- #
# Synchronized SDK comparison window
# --------------------------------------------------------------------- #


def run_compare(path: Path, args, assets_root: Path) -> int:
    """Run two independent scenes with identical settings and bounded teardown."""
    import multiprocessing as mp
    import queue

    if path.suffix == ".xml":
        raise ValueError("--compare requires a native model; MJCF remains adapter-only")
    ctx = mp.get_context("spawn")
    results, stop, barrier = ctx.Queue(), ctx.Event(), ctx.Barrier(2, timeout=20)
    settings = vars(args).copy()
    settings["model"] = str(path)
    processes = [
        ctx.Process(target=_compare_worker, args=(role, settings, results, stop, barrier))
        for role in ("adapter", "native")
    ]
    for process in processes:
        process.start()
    try:
        while any(process.is_alive() for process in processes):
            for process in processes:
                process.join(timeout=0.05)
            if any(p.exitcode not in (None, 0) for p in processes):
                stop.set()
                barrier.abort()
                break
    except KeyboardInterrupt:
        stop.set()
        barrier.abort()
    finally:
        stop.set()
        barrier.abort()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    records = []
    for _ in processes:
        try:
            records.append(results.get(timeout=1))
        except queue.Empty:
            break
    passed = len(records) == 2 and all(r["status"] == "passed" for r in records)
    passed &= all(p.exitcode == 0 for p in processes)
    report = {
        "status": "passed" if passed else "failed",
        "windows": records,
        "manual_inspection": "not claimed",
        "saved_images": False,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


def _compare_worker(role, settings, results, stop, barrier):
    """Each worker owns exactly one viewer and one simulation scene."""
    import threading
    from types import SimpleNamespace

    import superdex_compare as compare

    args = SimpleNamespace(**settings)
    path = Path(args.model)
    backend = reference = viewer = None
    held = False
    frames = 0
    status, error = "passed", None
    try:
        from unisim.backend.superdex.runtime import (
            acquire_runtime,
            load_superdex_dependencies,
            release_runtime,
        )

        physics, _ = load_superdex_dependencies()
        acquire_runtime(physics)
        held = True
        profile = registered_profile(path)
        config = (
            compare.load_controller_config(args.controller_config)
            if args.controller_config
            else None
        )
        target_spec = compare.load_absolute_target(args.absolute_target) if config else None
        options = backend_options(path, args, profile)
        if role == "adapter":
            from unisim import create_backend
            from unisim.scene import SceneCfg

            backend = create_backend(
                "superdex", SceneCfg(str(path), fragment_files=args.fragments), 1, SIM_DT, **options
            )
            if args.no_gravity:
                backend.set_gravity([0, 0, 0])
            names = list(backend.get_actuator_names())
            qi = backend.get_joint_state_qpos_indices(names)
            vi = backend.get_joint_state_qvel_indices(names)
            limits = backend.get_actuator_ctrl_range()[:, 1]
            armature = backend.get_dof_armature()[vi]
            q0 = backend.get_state()["qpos"][0, qi].copy()
            indices = [backend.get_model_info().coordinate_names.index(n) for n in names]
            ranges = backend.get_joint_range()[indices]
            if config:
                backend.configure_controller(**config)
                target = compare.shared_absolute_target(
                    config["type_name"], target_spec, backend.get_model_info()
                )
        else:
            if path.suffix in (".superdex_bot", ".superdex_bot_archive"):
                reference = compare.BotReference(path)
                actors = [reference.actor]
                names = list(reference.axis_columns)
                routes = [(0, int(i)) for i in reference.dofs]
                authored = np.asarray(reference.efforts)
                expanded = (
                    np.concatenate(
                        [
                            np.full(size, authored[i])
                            for i, (_, size) in enumerate(reference.column_sizes)
                        ]
                    )
                    if names
                    else np.zeros(0)
                )
                limits = np.broadcast_to(
                    options.get("superdex_effort_limits", expanded), (len(names),)
                ).copy()
                q0 = reference.joint_state()[0]
                ranges = reference.ranges
                armature = (
                    np.concatenate(
                        [
                            np.full(size, reference.armature[i])
                            for i, (_, size) in enumerate(reference.column_sizes)
                        ]
                    )
                    if names
                    else np.zeros(0)
                )
            else:
                reference = compare.SceneReference(path)
                actors = reference.articulated
                all_names, all_routes = compare.native_controls(actors)
                selected = options["superdex_controlled_joints"]
                names, routes = [], []
                for name in selected:
                    matches = [
                        i
                        for i, n in enumerate(all_names)
                        if n == name or n in [name + "/" + axis for axis in "xyz"]
                    ]
                    if not matches:
                        raise ValueError(f"unknown native controlled joint: {name}")
                    names.extend(all_names[i] for i in matches)
                    routes.extend(all_routes[i] for i in matches)
                limits = np.full(len(names), args.effort_limit or 1.0)
                q0 = np.array([compare.native_state(actors[ai])[0][d] for ai, d in routes])
                ranges = np.tile([-np.inf, np.inf], (len(names), 1))
                armature = np.ones(len(names)) * 0.001
            for fragment in args.fragments:
                loaded = physics.prefab.load_from_file(
                    fragment, str(compare._root_of(Path(fragment)))
                )
                physics.prefab.add_to_scene(loaded, reference.scene)
            if args.no_gravity:
                reference.scene.set_gravity([0, 0, 0])
            reference.scene.release_state(reference.snapshot)
            reference.snapshot = reference.scene.capture_state()
            if config:
                reference.configure(config)
                target = compare.absolute_target(reference, target_spec)
            from superdex.physics.viewer import Viewer, ViewerCfg

            viewer = Viewer(ViewerCfg(offscreen=args.frames is not None))
            viewer.set_scene(reference.scene)
            viewer.frame_scene()
        # Only registered bots use PD. Explicit scene controls are physical efforts.
        amplitudes = _sweep_amplitudes(q0, ranges, profile.sweep_amplitude) if profile else None
        kd = (
            2 * np.sqrt(profile.kp * np.maximum(armature, profile.armature_floor))
            if profile
            else None
        )
        reset_frame = args.frames * 2 // 3 if args.frames else int(8 / CTRL_DT)
        move_frame = args.frames // 3 if args.frames else int(2 / CTRL_DT)
        initial = backend.get_state() if backend else None
        reset_error = None

        def step_frame(obs=None):
            nonlocal frames, reset_error
            if stop.is_set():
                raise StopIteration
            if frames == reset_frame:
                if backend:
                    backend.reset()
                    reset_error = max(
                        compare.maximum(backend.get_state()[k] - value)
                        for k, value in initial.items()
                    )
                else:
                    reference.scene.restore_state(reference.snapshot, release_immediately=False)
                    if config:
                        reference.controller.reset()
                        if reference.kind == "MOCHI_ARTICULATED_POSE":
                            reference.controller.initialize(False)
            for _ in range(DECIMATION):
                if config:
                    if backend:
                        backend.step_controller([target])
                    else:
                        reference.step_controller(target, limits)
                    continue
                if profile and len(names):
                    target_q = q0 + (
                        amplitudes
                        * np.sin((frames - move_frame) * CTRL_DT * 2 * np.pi / profile.sweep_period)
                        if move_frame <= frames < reset_frame
                        else 0
                    )
                    if backend:
                        state = backend.get_state()
                        q, v = state["qpos"][0, qi], state["qvel"][0, vi]
                    else:
                        q, v = reference.joint_state()
                    command = np.clip(profile.kp * (target_q - q) - kd * v, -limits, limits)
                else:
                    command = np.zeros(len(names))
                    if args.controlled_joints and move_frame <= frames < reset_frame:
                        command = np.minimum(limits, 0.005) * np.sin(
                            frames * 0.013 + np.arange(len(names))
                        )
                if backend:
                    backend.step(command[None, :])
                else:
                    compare.apply_native_control(actors, routes, command)
                    reference.scene.step(SIM_DT)
            barrier.wait()
            if viewer:
                viewer.render()
            barrier.wait()
            frames += 1
            if viewer and viewer.user_requested_close():
                stop.set()
                barrier.abort()
            return obs

        if backend:
            backend.run_playback(
                env=None,
                initialize=lambda: None,
                step=step_frame,
                num_steps=args.frames,
                headless=args.frames is not None,
                record_video=False,
            )
        else:
            while not stop.is_set() and (args.frames is None or frames < args.frames):
                step_frame()
        if reset_error is not None and reset_error > RESET_TOLERANCE:
            raise RuntimeError(f"reset error {reset_error}")
    except (StopIteration, threading.BrokenBarrierError):
        if not stop.is_set():
            status, error = "failed", "comparison synchronization timed out"
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        barrier.abort()
        try:
            if viewer is not None:
                viewer.close()
            if reference is not None:
                reference.close()
            if backend is not None:
                backend.close()
        finally:
            if held:
                release_runtime(physics)
        results.put({"role": role, "status": status, "error": error, "frames": frames})


def run_single(path: Path, args, assets_root: Path) -> int:
    from unisim import create_backend
    from unisim.scene import SceneCfg

    kind = _kind_of(path)
    profile = registered_profile(path)
    kwargs = backend_options(path, args, profile)
    demo = None
    camera = None
    report = {"model": str(path), "format": kind, "sim_dt": SIM_DT}
    controller_label = args.controlled_joints
    from superdex_compare import (
        load_absolute_target,
        load_controller_config,
        shared_absolute_target,
    )

    config = load_controller_config(args.controller_config) if args.controller_config else None

    backend = create_backend(
        "superdex", SceneCfg(str(path), fragment_files=args.fragments), 1, SIM_DT, **kwargs
    )
    started = time.perf_counter()
    try:
        if args.no_gravity:
            backend.set_gravity([0.0, 0.0, 0.0])
        report["gravity"] = backend.get_gravity().tolist()

        # Controller-target mode for bots/archives.
        if args.controller_config and args.absolute_target:
            backend.configure_controller(**config)
            target = shared_absolute_target(
                config["type_name"],
                load_absolute_target(args.absolute_target),
                backend.get_model_info(),
            )
            kind_name = config["type_name"]
            report["controller"] = kind_name

            def step_controller_frame(obs):
                backend.step_controller([target], DECIMATION)
                return obs

            view = PassiveView(backend)
            view.step = step_controller_frame
            demo = None
        else:
            n = backend.num_actuators
            if n and profile is not None:
                names = backend.get_actuator_names()
                qi = backend.get_joint_state_qpos_indices(names)
                vi = backend.get_joint_state_qvel_indices(names)
                indices = [backend.get_model_info().coordinate_names.index(name) for name in names]
                amplitudes = _sweep_amplitudes(
                    backend.get_state()["qpos"][0, qi],
                    backend.get_joint_range()[indices],
                    profile.sweep_amplitude,
                )
                effort = backend.get_actuator_ctrl_range()[:, 1]
                kd = 2 * np.sqrt(
                    profile.kp * np.maximum(backend.get_dof_armature()[vi], profile.armature_floor)
                )
                demo = Demo(
                    backend,
                    kp=profile.kp,
                    kd=kd,
                    effort=effort,
                    amplitudes=amplitudes,
                    floating=False,
                    zero_gravity=args.no_gravity,
                )
                report["mode"] = "controlled"
                view = demo
            elif n and args.controlled_joints:
                view = PassiveView(backend)
                tick = 0

                def effort_step(obs):
                    nonlocal tick
                    command = np.minimum(backend.get_actuator_ctrl_range()[:, 1], 0.005) * np.sin(
                        tick * 0.013 + np.arange(n)
                    )
                    backend.step(command[None, :], DECIMATION)
                    tick += 1
                    return obs

                view.step = effort_step
                report["mode"] = "controlled-scene"
            else:
                demo = PassiveView(backend)
                report["mode"] = "passive"
                view = demo
            camera = compute_camera(backend)

        if args.frames is not None:
            if isinstance(view, Demo):
                global HOLD_SECONDS, MOVE_SECONDS, SETTLE_SECONDS
                HOLD_SECONDS = max(1, args.frames // 3) * CTRL_DT
                MOVE_SECONDS = max(1, args.frames // 3 - 1) * CTRL_DT
                SETTLE_SECONDS = CTRL_DT
            else:
                original_step = view.step
                frame = 0
                initial = backend.get_state()

                def bounded_step(obs):
                    nonlocal frame
                    if frame == args.frames * 2 // 3:
                        backend.reset()
                        report["reset_error"] = max(
                            float(np.max(np.abs(backend.get_state()[k] - v), initial=0))
                            for k, v in initial.items()
                        )
                    frame += 1
                    return original_step(obs)

                view.step = bounded_step
        interrupted = False
        try:
            backend.run_playback(
                env=None,
                initialize=lambda: None,
                step=view.step,
                num_steps=args.frames,
                headless=args.frames is not None,
                record_video=False,
                camera_kwargs=camera,
            )
        except KeyboardInterrupt:
            interrupted = True
        finally:
            report.update(view.evidence())
            report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            report["viewer"] = "offscreen-smoke" if args.frames is not None else "interactive"
            if controller_label:
                report["controlled_joints"] = controller_label
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"report: {args.out}")
            else:
                print(json.dumps(report, indent=2))
        if report.get("reset_error", 0) > RESET_TOLERANCE:
            raise RuntimeError(f"reset error {report['reset_error']}")
        if interrupted:
            print("interrupted: viewer and backend released")
            return 0
        if isinstance(demo, Demo) and demo.reset_error is None and args.frames is None:
            print("note: the reset phase never ran (window closed early)")
        if (
            isinstance(demo, Demo)
            and demo.reset_error is not None
            and demo.reset_error > RESET_TOLERANCE
        ):
            print(
                f"error: reset did not restore the default pose ({demo.reset_error:.2e})",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "model",
        nargs="?",
        type=str,
        default=None,
        help="model path (absolute, or relative to the asset root)",
    )
    parser.add_argument(
        "--assets", type=str, default=None, help="asset bundle root (default $SUPERDEX_ASSETS_PATH)"
    )
    parser.add_argument(
        "--list-profiles", action="store_true", help="list registered bot profiles and exit"
    )
    parser.add_argument(
        "--controlled-joints",
        type=str,
        default=None,
        help="ordered controlled joint list for scenes/prefabs",
    )
    parser.add_argument(
        "--effort-limit",
        type=float,
        default=None,
        help="scalar effort limit (required with --controlled-joints)",
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=None,
        help="SDK controller JSON (type_name/param_args/init_args)",
    )
    parser.add_argument(
        "--absolute-target",
        type=Path,
        default=None,
        help="absolute controller target JSON for --controller-config",
    )
    parser.add_argument(
        "--no-gravity",
        action="store_true",
        help="disable gravity for this session (resets keep it)",
    )
    parser.add_argument(
        "--fragments",
        nargs="*",
        default=[],
        help="rigid prefab fragments composed with a native bot",
    )
    parser.add_argument(
        "--compare", action="store_true", help="open synchronized adapter and direct-SDK windows"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="headless smoke run: render N frames, no window"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional JSON evidence report path (nothing is written by default)",
    )
    args = parser.parse_args(argv)

    if args.list_profiles:
        for key, profile in sorted(PROFILES.items()):
            print(f"{key:28s} {profile.relpath}")
        return 0

    if not args.model:
        parser.error("pass a model path (or --list-profiles)")

    if args.frames is not None and args.frames < 6:
        parser.error("--frames must be at least 6 to exercise movement and reset")
    if bool(args.controller_config) != bool(args.absolute_target):
        parser.error("--controller-config and --absolute-target must be supplied together")
    if args.effort_limit is not None and (
        not np.isfinite(args.effort_limit) or args.effort_limit <= 0
    ):
        parser.error("--effort-limit must be finite and positive")
    assets_root = (
        Path(
            args.assets
            or os.environ.get("SUPERDEX_ASSETS_PATH", REPOSITORY_ROOT / "assets/superdex")
        )
        .expanduser()
        .resolve()
    )
    path = resolve_model(args.model, assets_root)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise actionable_format_error(path)

    if args.controlled_joints and path.suffix not in (".mochi_scene", ".mochi_prefab"):
        parser.error("--controlled-joints applies only to scenes/prefabs")
    if args.controlled_joints and args.effort_limit is None:
        parser.error("--controlled-joints requires --effort-limit")
    if (args.controller_config or args.fragments) and path.suffix not in (
        ".superdex_bot",
        ".superdex_bot_archive",
    ):
        parser.error("controllers and fragments require a native bot/archive")
    args.fragments = [str(resolve_model(p, assets_root)) for p in args.fragments]
    # Scene dependencies without a ./ prefix are asset-root relative. Use the
    # same root for model selection, adapter loading and spawned SDK viewers.
    previous_root = os.environ.get("SUPERDEX_ASSETS_PATH")
    dependency_root = assets_root
    if previous_root is None and args.assets is None and not path.is_relative_to(assets_root):
        # An absolute external model without a root marker keeps its local
        # dependency fallback instead of inheriting the repository bundle.
        dependency_root = path.parent
    os.environ["SUPERDEX_ASSETS_PATH"] = str(dependency_root)
    try:
        if args.compare:
            return run_compare(path, args, assets_root)
        return run_single(path, args, assets_root)
    finally:
        if previous_root is None:
            os.environ.pop("SUPERDEX_ASSETS_PATH", None)
        else:
            os.environ["SUPERDEX_ASSETS_PATH"] = previous_root


if __name__ == "__main__":
    raise SystemExit(main())
