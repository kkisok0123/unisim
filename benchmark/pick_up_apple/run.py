#!/usr/bin/env python3
"""Run Dexlab's physical apple-stem / fruit grasp through UniSim SuperDex."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from unisim import ArticulationPoseTarget, create_backend
from unisim.scene import SceneCfg

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ROBOT = ROOT / "assets/robot/openarm_v20_wuji_trimmed.superdex_bot"
OBJECTS = ROOT / "assets/objects/objects.mochi_prefab"
DT = 0.002
SAMPLE_DT = 0.05
STEPS_PER_SAMPLE = round(SAMPLE_DT / DT)


def build_controller_params(joints, links, *, body_grasp=False):
    """Controller arrays follow the loaded prefab, never a recorded frame order."""
    tracking = []
    for joint in joints:
        if joint.type.name != "REVOLUTE":
            gains = (0, 0)
        elif joint.name.startswith(("l_", "r_")):
            stiffness = 30 if joint.name.startswith(("r_index_finger", "r_thumb")) else 0.8
            gains = (stiffness, 0.02)
            if body_grasp and joint.name.startswith("r_"):
                gains = (0.8, 0.01)
        else:
            gains = (1000, 40)
        tracking.append(dict(stiffness=gains[0], damping=gains[1], saturation=-1))
    zero = [dict(stiffness=0, damping=0, saturation=-1) for _ in links]
    return {
        "poseControllerParams": dict(
            jointTracking=tracking, linkPosTracking=zero, linkRotTracking=zero
        )
    }


def materialize_robot(directory):
    """Resolve repository-local dependencies without changing the qualified asset bundle."""
    directory = Path(directory)
    (directory / ".superdex_root").touch()
    bundle = (
        Path(os.environ.get("SUPERDEX_ASSETS_PATH", REPO / "assets/superdex"))
        .expanduser()
        .resolve()
        / "bots"
    )
    for name in ("arms", "hands"):
        (directory / name).symlink_to(bundle / name, target_is_directory=True)
    for source in ROBOT.parent.iterdir():
        if source != ROBOT:
            (directory / source.name).symlink_to(source)
    data = json.loads(ROBOT.read_text())
    path = directory / ROBOT.name
    path.write_text(json.dumps(data))
    return path


def make_backend(robot):
    return create_backend(
        "superdex",
        SceneCfg(str(robot), fragment_files=[str(OBJECTS)]),
        num_envs=1,
        sim_dt=DT,
        superdex_execution_mode="serial",
        superdex_num_worker_threads=8,
    )


def run_task(backend, seconds, *, robot=ROBOT, viewer=False, offscreen=False, output=None):
    import trimesh
    from native import Instrumentation, planning_robot, settled_apple
    from planning import GraspPlan, plan_body_grasp
    from verify import verify_sequence

    backend.reset()
    body_mesh = trimesh.load_mesh(ROOT / "assets/objects/apple-collision.obj", process=False)
    stem_mesh = trimesh.load_mesh(ROOT / "assets/objects/stem-collision.obj", process=False)
    print("Settling the apple, then planning the stem pinch", flush=True)
    apple_q, apple_v = settled_apple(backend, DT)
    info = backend.get_model_info()
    apple_layout = backend.get_root_state_layout("apple_with_stem")
    records, dynamics, frames = [], [], []
    count = 0
    total_steps = round(seconds / DT)
    state_finite = True
    with planning_robot(robot) as (prefab, context, actor, links):
        plan = GraspPlan(
            prefab, actor, links, apple_q, json.loads((ROOT / "assets/stem.json").read_text())
        )
        if tuple(plan.joint_names) != info.coordinate_names:
            raise ValueError("planner and backend joint order differ")
        native_names = [link.name for link in prefab.links]
        description = next(
            d
            for d in backend.get_controller_descriptions()
            if d.identifier == "MOCHI_ARTICULATED_POSE"
        )
        if tuple(native_names) != description.link_names:
            raise ValueError("planner and controller link order differ")
        state = backend.get_state()
        state["qpos"][0, list(info.coordinate_qpos_indices)] = plan.pre
        state["qvel"][:] = 0
        state["qpos"][0, apple_layout.qpos_indices] = apple_q
        state["qvel"][0, apple_layout.qvel_indices] = apple_v
        backend.set_state(np.array([0]), state["qpos"], state["qvel"])
        instrumentation = Instrumentation(backend, body_mesh, stem_mesh, prefab)
        robot_body_ids = list(info.articulations[0].body_ids)
        all_ids = robot_body_ids + list(backend.get_body_ids(["apple_with_stem"]))
        base_id = int(backend.get_body_ids(["openarm_body_link0"])[0])
        base_pos, base_quat = backend.get_body_pose_w([base_id])
        base_pose = np.r_[base_pos[0, 0], base_quat[0, 0]]
        max_base_change = 0.0
        # The HARD root joint authors the placement; the actor frame remains identity.
        root = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        def configure(body_grasp=False):
            backend.configure_controller(
                "MOCHI_ARTICULATED_POSE",
                param_args=json.dumps(
                    build_controller_params(prefab.joints, prefab.links, body_grasp=body_grasp)
                ),
                init_args="{}",
            )

        configure()

        def advance(_):
            nonlocal count, state_finite, max_base_change
            for _ in range(min(STEPS_PER_SAMPLE, total_steps - count)):
                t = count * DT
                if count == round(23 / DT):
                    # Replan against the actual released fruit, not historical evidence.
                    current = backend.get_state()
                    plan.regrasp = plan_body_grasp(
                        prefab,
                        context,
                        backend.get_dof_pos()[0].copy(),
                        plan.arm,
                        plan.hand,
                        plan.bounds,
                        current["qpos"][0, apple_layout.qpos_indices][:3],
                    )
                    backend.clear_controller()
                    configure(body_grasp=True)
                target = ArticulationPoseTarget(root, joint_positions=plan.target(t))
                previous_velocity = instrumentation.velocity()
                backend.step_controller([target], nsteps=1)
                row = instrumentation.dynamics(t, previous_velocity)
                dynamics.append(row)
                current = backend.get_state()
                state_finite &= bool(
                    np.isfinite(current["qpos"]).all() and np.isfinite(current["qvel"]).all()
                )
                if not state_finite:
                    raise RuntimeError(f"Non-finite task state at {t:.3f}s")
                if count % STEPS_PER_SAMPLE == 0:
                    pose = current["qpos"][0, apple_layout.qpos_indices]
                    records.append(instrumentation.metrics(t, pose))
                    pos, quat = backend.get_body_pose_w(all_ids)
                    # Retain Dexlab's xyz + xyzw recording format for independent verification.
                    frames.append(np.c_[pos[0], quat[0][:, [1, 2, 3, 0]]])
                    pos, quat = backend.get_body_pose_w([base_id])
                    max_base_change = max(
                        max_base_change,
                        float(np.max(np.abs(np.r_[pos[0, 0], quat[0, 0]] - base_pose))),
                    )
                if count % round(1 / DT) == 0:
                    print(
                        f"{t:.1f}s clearance={records[-1]['clearance']:.4f} "
                        f"solver={row['scene_status']}",
                        flush=True,
                    )
                count += 1
            return None

        try:
            if viewer:
                backend.run_playback(
                    env=SimpleNamespace(cfg=SimpleNamespace(ctrl_dt=SAMPLE_DT)),
                    initialize=lambda: None,
                    step=advance,
                    num_steps=(total_steps + STEPS_PER_SAMPLE - 1) // STEPS_PER_SAMPLE,
                    headless=offscreen,
                    record_video=False,
                    camera_kwargs=dict(
                        cam_lookat=[0.1, 0.22, 0.5],
                        cam_distance=2.4,
                        cam_azimuth=225,
                        cam_elevation=-23,
                    ),
                )
            else:
                while count < total_steps:
                    advance(None)
        finally:
            backend.clear_controller()
    complete = count >= round(40 / DT)
    if complete:
        summary = verify_sequence(records, dynamics, DT)
    else:
        summary = dict(checks={}, passed=False)
    summary.update(
        physics="UniSim SuperDex FP64",
        physics_dt=DT,
        physics_steps=count,
        seconds=count * DT,
        requested_seconds=seconds,
        complete_sequence=complete,
    )
    summary["checks"].update(
        base_stationary=max_base_change < 1e-12,
        all_recorded_transforms_finite=bool(np.isfinite(frames).all()),
        finite_state=state_finite,
        requested_steps_completed=count == total_steps,
    )
    summary["passed"] = bool(complete and all(summary["checks"].values()))
    summary["maximum_base_pose_change"] = max_base_change
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(records) + "\n")
        (output / "dynamics.json").write_text(json.dumps(dynamics) + "\n")
        np.savez_compressed(
            output / "trajectory.npz", frames=frames, names=native_names + ["apple"], dt=SAMPLE_DT
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--offscreen", action="store_true", help="Render viewer smoke frames offscreen"
    )
    parser.add_argument("--out", type=Path, help="Optional JSON summary file")
    parser.add_argument(
        "--output", type=Path, help="Optional directory for detailed verification data"
    )
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not DT <= args.seconds <= 40:
        parser.error("--seconds must be between 0.002 and 40")
    os.environ["SUPERDEX_PRECISION"] = "fp64"
    os.environ.setdefault("SUPERDEX_ASSETS_PATH", str(REPO / "assets/superdex"))
    from superdex import physics

    if not physics.uses_double_precision():
        raise RuntimeError("Apple stem grasp requires a fresh process using SuperDex FP64")
    with TemporaryDirectory(prefix="unisim-apple-") as directory:
        robot = materialize_robot(directory)
        print("Building the FP64 robot and 0.2 mm apple collision grid", flush=True)
        backend = make_backend(robot)
        try:
            result = run_task(
                backend,
                args.seconds,
                robot=robot,
                viewer=args.viewer or args.offscreen,
                offscreen=args.offscreen,
                output=args.output,
            )
        finally:
            backend.close()
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.seconds == 40 and not result["passed"]:
        return 1
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
