#!/usr/bin/env python3
"""Run the apple grasp through the UniSim MuJoCo adapter."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from planning import GraspPlan, Kinematics, plan_body_grasp
from runtime import (
    APPLE_SDF_GEOM,
    DISCRETE_AVAILABLE,
    DT,
    GAIN_SWITCH_TIME,
    INTEGRATOR,
    ROOT,
    build_settling_scene,
    compile_physics_model,
    controller_metadata,
    hinge_names,
    sdf_collision_metadata,
    set_position_gains,
    write_runtime_scene,
)

from unisim import create_backend
from unisim.scene import SceneCfg

SAMPLE_DT = 0.05
STEPS_PER_SAMPLE = round(SAMPLE_DT / DT)
TABLE_TOP_Z = 0.2995


def model_joint_indices(model):
    names = hinge_names(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qpos = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    qvel = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
    return names, qpos, qvel


def _mesh_vertices(path: Path) -> np.ndarray:
    return np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text(encoding="ascii").splitlines()
            if line.startswith("v ")
        ],
        dtype=float,
    )


def apple_clearance(
    position: np.ndarray, quaternion_wxyz: np.ndarray, vertices: np.ndarray
) -> float:
    """Return the lowest apple-body vertex above the table surface."""
    vector = np.asarray(quaternion_wxyz[1:], dtype=float)
    scalar = float(quaternion_wxyz[0])
    rotated = vertices + 2 * np.cross(vector, np.cross(vector, vertices) + scalar * vertices)
    return float(np.min(rotated[:, 2] + position[2]) - TABLE_TOP_Z)


def launch_live_viewer(model):
    """Open an independent passive viewer for adapter-owned MuJoCo state."""
    import mujoco.viewer

    data = mujoco.MjData(model)
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except Exception as error:
        raise RuntimeError(
            "MuJoCo viewer startup failed; confirm that a graphical display is available"
        ) from error
    viewer.cam.lookat[:] = [0.25, 0.30, 0.42]
    viewer.cam.distance = 1.35
    viewer.cam.azimuth = 225
    viewer.cam.elevation = -23
    return viewer, data


def load_live_viewer_model(runtime_scene: Path, backend_model):
    """Load the unstripped MJCF used only for passive visualization."""
    model = mujoco.MjModel.from_xml_path(str(runtime_scene))
    if model.nq != backend_model.nq or model.nv != backend_model.nv:
        raise ValueError(
            "Viewer and adapter models have different state dimensions: "
            f"qpos {model.nq}/{backend_model.nq}, qvel {model.nv}/{backend_model.nv}"
        )
    return model


def load_collision_viewer_model(
    backend_model: mujoco.MjModel, collision_metadata: dict[str, object]
) -> mujoco.MjModel:
    """Copy the compiled physics model without recompiling its SDF grid."""
    model = copy.copy(backend_model)
    octree_fields = ("oct_aabb", "oct_child", "oct_coeff", "oct_depth")
    if any(
        not np.array_equal(getattr(model, field), getattr(backend_model, field))
        for field in octree_fields
    ) or sdf_collision_metadata(model) != collision_metadata:
        raise RuntimeError("Viewer model differs from the adapter's compiled SDF model")
    return model


def native_state_from_backend(backend) -> tuple[np.ndarray, np.ndarray]:
    """Return the single environment's native MuJoCo qpos and qvel."""
    if not hasattr(backend, "get_physics_state"):
        # Keep the settling-window logic usable with a minimal backend test
        # double; real MuJoCo backends always take the native-state branch.
        state = backend.get_state()
        return state["qpos"][0], state["qvel"][0]
    state = backend.get_physics_state()[0]
    qpos_start = 1
    qvel_start = qpos_start + backend.model.nq
    return (
        state[qpos_start:qvel_start],
        state[qvel_start : qvel_start + backend.model.nv],
    )


def body_poses_from_backend(backend, data, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate body poses from adapter-owned state without tracking sensors."""
    qpos, qvel = native_state_from_backend(backend)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(backend.model, data)
    return data.xpos[body_ids][None].copy(), data.xquat[body_ids][None].copy()


def sync_live_viewer(backend, viewer_model, viewer, data) -> None:
    """Copy the current single-environment backend state into the viewer."""
    qpos = backend.get_dof_pos()[0]
    qvel = backend.get_dof_vel()[0]
    if qpos.shape != data.qpos.shape or qvel.shape != data.qvel.shape:
        raise ValueError(
            "MuJoCo viewer state shape differs from the adapter model: "
            f"qpos {qpos.shape}/{data.qpos.shape}, qvel {qvel.shape}/{data.qvel.shape}"
        )
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(viewer_model, data)
    viewer.sync()


SETTLE_MIN_SECONDS = 3.0
SETTLE_WINDOW_SECONDS = 0.5
SETTLE_TIMEOUT_SECONDS = 10.0
SETTLE_POSITION_TOLERANCE = 0.00025  # Allows measured mesh-contact jitter (0.25 mm).
SETTLE_ANGLE_TOLERANCE = np.deg2rad(0.25)


def settling_pose_excursion(poses: np.ndarray) -> tuple[float, float]:
    """Maximum displacement/rotation from the first pose in a time window."""
    quaternions = poses[:, 3:7]
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    # q and -q represent the same rotation.
    dots = np.clip(np.abs(quaternions @ quaternions[0]), 0.0, 1.0)
    return (
        float(np.max(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1))),
        float(np.max(2 * np.arccos(dots))),
    )


def settled_apple(scene: Path, dt: float, *, diagnostics: dict | None = None):
    """Wait for sustained pose stability in an isolated apple/table scene."""
    model, collision_metadata = compile_physics_model(scene)
    backend = create_backend(
        "mujoco", SceneCfg(model), num_envs=1, sim_dt=dt, np_dtype=np.float64
    )
    try:
        backend.materialize()
        backend.reset()
        joint = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, "apple_with_stem_root")
        qadr = int(backend.model.jnt_qposadr[joint])
        vadr = int(backend.model.jnt_dofadr[joint])
        window_size = int(np.ceil(SETTLE_WINDOW_SECONDS / dt)) + 1
        poses = deque(maxlen=window_size)
        position_excursion = angle_excursion = float("inf")
        for steps in range(1, int(np.floor(SETTLE_TIMEOUT_SECONDS / dt)) + 1):
            backend.step(np.zeros((1, backend.num_actuators)), nsteps=1)
            native_qpos, native_qvel = native_state_from_backend(backend)
            qpos = native_qpos[qadr : qadr + 7].copy()
            qvel = native_qvel[vadr : vadr + 6].copy()
            if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
                raise RuntimeError("Apple settling produced a non-finite state")
            if np.linalg.norm(qpos[3:]) < 1e-12:
                raise RuntimeError("Apple settling produced an invalid quaternion")
            poses.append(qpos)
            if steps * dt < SETTLE_MIN_SECONDS or len(poses) < window_size:
                continue
            position_excursion, angle_excursion = settling_pose_excursion(np.asarray(poses))
            if (
                position_excursion <= SETTLE_POSITION_TOLERANCE
                and angle_excursion <= SETTLE_ANGLE_TOLERANCE
            ):
                if diagnostics is not None:
                    diagnostics.update(
                        seconds=steps * dt,
                        minimum_seconds=SETTLE_MIN_SECONDS,
                        window_seconds=(window_size - 1) * dt,
                        position_tolerance_m=SETTLE_POSITION_TOLERANCE,
                        angle_tolerance_rad=float(SETTLE_ANGLE_TOLERANCE),
                        position_excursion_m=position_excursion,
                        angle_excursion_rad=angle_excursion,
                        final_linear_speed_m_s=float(np.linalg.norm(qvel[:3])),
                        final_angular_speed_rad_s=float(np.linalg.norm(qvel[3:])),
                        collision_model=collision_metadata,
                    )
                # Preserve physical velocities rather than freezing the apple.
                return qpos, qvel
        raise RuntimeError(
            f"Apple did not sustain pose stability within {SETTLE_TIMEOUT_SECONDS:g}s: "
            f"position excursion={position_excursion:.6g}m "
            f"(limit {SETTLE_POSITION_TOLERANCE:g}m), "
            f"rotation excursion={np.rad2deg(angle_excursion):.6g}deg "
            f"(limit {np.rad2deg(SETTLE_ANGLE_TOLERANCE):g}deg)"
        )
    finally:
        backend.close()


def _phase_summary(records: list[dict[str, float]]) -> tuple[dict, dict]:
    phases = {}
    checks = {}
    definitions = {
        "stem_hold": (11.0, 14.0),
        "released": (22.0, 23.0),
        "body_hold": (37.0, 40.0),
    }
    for name, (start, end) in definitions.items():
        rows = [row for row in records if start <= row["time"] < end]
        if not rows:
            checks[f"{name}_recorded"] = False
            continue
        minimum = min(row["clearance"] for row in rows)
        maximum = max(row["clearance"] for row in rows)
        phases[name] = {
            "minimum_clearance_m": minimum,
            "maximum_clearance_m": maximum,
        }
        if name == "released":
            checks["released_on_table"] = maximum < 0.001
        else:
            checks[f"{name}_lifted"] = minimum > 0.07
    return phases, checks


def run_task(
    runtime_scene: Path,
    seconds: float,
    *,
    viewer: bool = False,
    viewer_visual: bool = False,
    output: Path | None = None,
    contacts_enabled: bool = True,
):
    del output
    settling_scene = runtime_scene.parent / "settling.xml"
    settling_diagnostics: dict = {}
    apple_q, apple_v = settled_apple(settling_scene, DT, diagnostics=settling_diagnostics)
    physics_model, collision_metadata = compile_physics_model(runtime_scene)
    backend = create_backend(
        "mujoco",
        SceneCfg(physics_model),
        num_envs=1,
        sim_dt=DT,
        np_dtype=np.float64,
        base_name="openarm_body_link0",
    )
    live_viewer = None
    previous_warning_handler = mujoco.get_mju_user_warning()
    runtime_warnings: list[str] = []

    def capture_warning(message: str) -> None:
        runtime_warnings.append(str(message))

    def is_bad_state_warning(message: str) -> bool:
        upper = message.upper()
        return any(
            marker in upper
            for marker in ("BADQ", "QACC", "QVEL", "QPOS", "NAN", "INF", "HUGE")
        )

    try:
        mujoco.set_mju_user_warning(capture_warning)
        backend.materialize()
        backend.reset()
        if backend.model.opt.timestep != DT:
            raise RuntimeError(f"MuJoCo runtime timestep is not {DT}")
        if int(backend.model.opt.integrator) != int(INTEGRATOR):
            raise RuntimeError("MuJoCo runtime integrator differs from controller metadata")
        if not (
            backend.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        ):
            raise RuntimeError("MuJoCo automatic bad-state reset is still enabled")
        names, qpos_ids, _ = model_joint_indices(backend.model)
        kin = Kinematics()
        if names != kin.names:
            raise ValueError("planner and MuJoCo adapter joint order differ")
        stem = json.loads((ROOT / "superdex/assets/stem.json").read_text())
        plan = GraspPlan(kin, apple_q, stem)
        qpos = np.asarray(backend.model.qpos0, dtype=float).copy()
        qvel = np.zeros(backend.model.nv, dtype=float)
        qpos[qpos_ids] = plan.pre
        apple_layout = backend.get_root_state_layout("apple_with_stem")
        qpos[list(apple_layout.qpos_indices)] = apple_q
        qvel[list(apple_layout.qvel_indices)] = apple_v
        backend.set_state(np.asarray([0]), qpos[None, :], qvel[None, :])

        _, _, gain_model_count = set_position_gains(backend, names)
        pose_data = mujoco.MjData(backend.model)
        base_id = int(backend.get_body_ids(["openarm_body_link0"])[0])
        apple_id = int(backend.get_body_ids(["apple_with_stem"])[0])
        base_pos, base_quat = body_poses_from_backend(
            backend, pose_data, np.asarray([base_id], dtype=int)
        )
        base_pose = np.r_[base_pos[0, 0], base_quat[0, 0]]
        apple_vertices = _mesh_vertices(ROOT / "superdex/assets/objects/apple-collision.obj")
        body_ids = np.arange(1, backend.model.nbody, dtype=int)
        body_names = [
            mujoco.mj_id2name(backend.model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in body_ids
        ]
        records, frames = [], []
        total_steps = round(seconds / DT)
        steps_completed = 0
        state_finite = True
        monotonic_time = True
        max_base_change = 0.0
        max_tracking_error = 0.0
        gain_switch_applied = seconds <= GAIN_SWITCH_TIME
        failure_reason = None
        viewer_started = None
        viewer_model = None
        previous_time = float(backend.get_physics_state()[0, 0])

        if viewer:
            if viewer_visual:
                # The visual scene is a separate XML compilation, not the
                # adapter's high-resolution physics model.
                viewer_model = load_live_viewer_model(runtime_scene, backend.model)
            else:
                viewer_model = load_collision_viewer_model(backend.model, collision_metadata)
                print(
                    "Viewer displays a copy of the adapter's compiled collision model "
                    f"({APPLE_SDF_GEOM}, SDF depth {collision_metadata['octree_depth']}).",
                    flush=True,
                )
            live_viewer, viewer_data = launch_live_viewer(viewer_model)
            sync_live_viewer(backend, viewer_model, live_viewer, viewer_data)
            # Give Wayland/GLFW time to present the initialized apple before
            # simulation starts (and before an early hard-failure can close it).
            time.sleep(1.0)
            viewer_started = time.monotonic()
        for count in range(total_steps):
            if live_viewer is not None and count % STEPS_PER_SAMPLE == 0:
                if not live_viewer.is_running():
                    break
            t = count * DT
            if count == round(GAIN_SWITCH_TIME / DT):
                # Replan against the actual released fruit, then switch all
                # right-hand gains on host and pool models between steps.
                apple_pos, _ = body_poses_from_backend(
                    backend, pose_data, np.asarray([apple_id], dtype=int)
                )
                plan.regrasp = plan_body_grasp(
                    kin,
                    backend.get_dof_pos()[0, qpos_ids].copy(),
                    apple_pos[0, 0].copy(),
                )
                _, _, switched_model_count = set_position_gains(
                    backend, names, body_grasp=True
                )
                if switched_model_count != gain_model_count:
                    raise RuntimeError("MuJoCo model pool changed during the gain update")
                gain_switch_applied = True
            target = plan.target(t)
            q = backend.get_dof_pos()[0, qpos_ids]
            tracking_error = float(np.max(np.abs(target - q)))
            max_tracking_error = max(max_tracking_error, tracking_error)
            backend.step(target[None, :], nsteps=1)
            steps_completed = count + 1
            current_time = float(backend.get_physics_state()[0, 0])
            expected_time = steps_completed * DT
            time_valid = current_time > previous_time and np.isclose(
                current_time, expected_time, rtol=0, atol=1e-9
            )
            monotonic_time &= bool(time_valid)
            if not time_valid:
                failure_reason = (
                    "MuJoCo simulation time reset or skipped: "
                    f"expected {expected_time:.6f}, got {current_time:.6f}"
                )
                break
            previous_time = current_time
            current_qpos, current_qvel = native_state_from_backend(backend)
            state_finite &= bool(
                np.isfinite(current_qpos).all() and np.isfinite(current_qvel).all()
            )
            bad_state_warnings = [
                warning for warning in runtime_warnings if is_bad_state_warning(warning)
            ]
            if not state_finite or bad_state_warnings:
                detail = bad_state_warnings[-1] if bad_state_warnings else "non-finite state"
                failure_reason = f"MuJoCo solver divergence at {t:.3f}s: {detail}"
                break
            if count % STEPS_PER_SAMPLE == 0:
                apple_pos, apple_quat = body_poses_from_backend(
                    backend, pose_data, np.asarray([apple_id], dtype=int)
                )
                clearance = apple_clearance(
                    apple_pos[0, 0], apple_quat[0, 0], apple_vertices
                )
                records.append(
                    {
                        "time": t,
                        "apple_z": float(apple_pos[0, 0, 2]),
                        "clearance": clearance,
                        "max_joint_tracking_error": tracking_error,
                    }
                )
                pos, quat = body_poses_from_backend(backend, pose_data, body_ids)
                frames.append(np.c_[pos[0], quat[0][:, [1, 2, 3, 0]]])
                pos, quat = body_poses_from_backend(
                    backend, pose_data, np.asarray([base_id], dtype=int)
                )
                max_base_change = max(
                    max_base_change,
                    float(np.max(np.abs(np.r_[pos[0, 0], quat[0, 0]] - base_pose))),
                )
            if live_viewer is not None and steps_completed % STEPS_PER_SAMPLE == 0:
                assert viewer_started is not None
                delay = viewer_started + steps_completed * DT - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if live_viewer.is_running():
                    assert viewer_model is not None
                    sync_live_viewer(backend, viewer_model, live_viewer, viewer_data)
            if count % round(1 / DT) == 0:
                print(
                    f"{t:.1f}s apple_z={records[-1]['apple_z']:.4f} "
                    f"clearance={records[-1]['clearance']:.4f}",
                    flush=True,
                )

        frames_array = np.asarray(frames)
        checks = {
            "finite_state": state_finite,
            "no_bad_state_warning": not any(
                is_bad_state_warning(warning) for warning in runtime_warnings
            ),
            "monotonic_simulation_time": monotonic_time,
            "base_stationary": max_base_change < 1e-12,
            "all_recorded_transforms_finite": bool(np.isfinite(frames_array).all()),
            "requested_steps_completed": steps_completed == total_steps,
            "gain_switch_applied": gain_switch_applied,
        }
        phases = {}
        complete = steps_completed >= round(40 / DT)
        if contacts_enabled and seconds >= 9:
            lift_rows = [row for row in records if 8 <= row["time"] < 9]
            checks["nine_second_apple_lifted"] = bool(lift_rows) and max(
                row["clearance"] for row in lift_rows
            ) > 0.01
        if complete:
            phases, phase_checks = _phase_summary(records)
            checks.update(phase_checks)
        summary = {
            "physics": "UniSim MuJoCo",
            "physics_dt": DT,
            "control_dt": DT,
            "physics_steps": steps_completed,
            "seconds": steps_completed * DT,
            "requested_seconds": seconds,
            "complete_sequence": complete,
            "contacts_enabled": contacts_enabled,
            "controller": controller_metadata(),
            "collision_model": collision_metadata,
            "settling": settling_diagnostics,
            "gain_model_count": gain_model_count,
            "checks": checks,
            "phases": phases,
            "maximum_joint_tracking_error": max_tracking_error,
            "maximum_base_pose_change": max_base_change,
            "solver_warnings": [
                warning for warning in runtime_warnings if is_bad_state_warning(warning)
            ],
            "viewer_warnings": [
                warning for warning in runtime_warnings if not is_bad_state_warning(warning)
            ],
            "failure_reason": failure_reason,
        }
        summary["passed"] = bool(all(checks.values()))
        return summary, records, frames_array, body_names
    finally:
        mujoco.set_mju_user_warning(previous_warning_handler)
        if live_viewer is not None:
            live_viewer.close()
            deadline = time.monotonic() + 2.0
            while live_viewer.is_running() and time.monotonic() < deadline:
                time.sleep(0.01)
            # exitrequest becomes visible before the render thread has finished
            # destroying its GLFW window; release viewer references first.
            live_viewer = None
            viewer_data = None
            viewer_model = None
            gc.collect()
            time.sleep(2.0)
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument(
        "--viewer", action="store_true", help="Show the actual compiled collision model"
    )
    parser.add_argument(
        "--viewer-visual", action="store_true", help="Show the separate visual-only model"
    )
    parser.add_argument(
        "--disable-contacts",
        action="store_true",
        help="Disable contacts for the controller-only stability gate",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not DT <= args.seconds <= 40:
        parser.error("--seconds must be between 0.002 and 40")
    runtime = write_runtime_scene(disable_contacts=args.disable_contacts)
    build_settling_scene(runtime.parent / "settling.xml")
    if not DISCRETE_AVAILABLE:
        print(
            "NOTICE: the requested discrete integrator was introduced in MuJoCo 3.13, "
            "but this project is pinned to MuJoCo 3.11 by mujoco-uni-runtime 0.5.0; "
            "using implicit and recording the unsupported capability.",
            flush=True,
        )
    try:
        summary, records, frames, names = run_task(
            runtime,
            args.seconds,
            viewer=args.viewer or args.viewer_visual,
            viewer_visual=args.viewer_visual,
            output=args.output,
            contacts_enabled=not args.disable_contacts,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; viewer and MuJoCo backend closed.", flush=True)
        return 130
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "metrics.json").write_text(json.dumps(records) + "\n")
        np.savez_compressed(
            args.output / "trajectory.npz", frames=frames, names=names, dt=SAMPLE_DT
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
