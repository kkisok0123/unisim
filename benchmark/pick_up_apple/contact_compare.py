#!/usr/bin/env python3
"""Record comparable apple/pad and apple/table contact diagnostics by engine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PAIRS = (
    "index_stem",
    "thumb_stem",
    "index_fruit",
    "thumb_fruit",
    "apple_table",
    "other_apple_contact",
)
SAMPLE_DT = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_pair() -> dict:
    return {
        "count": 0,
        "force": np.zeros(3),
        "normal": np.zeros(3),
        "friction": np.zeros(3),
        "normal_load": 0.0,
        "friction_load": 0.0,
        "max_depth": 0.0,
        "depth_sum": 0.0,
        "position_sum": np.zeros(3),
        "negative_normal_count": 0,
    }


def _add_contact(
    pair: dict,
    force: np.ndarray,
    normal_force: np.ndarray,
    normal_load: float,
    depth: float,
    position_world: np.ndarray,
) -> None:
    friction_force = force - normal_force
    pair["count"] += 1
    pair["force"] += force
    pair["normal"] += normal_force
    pair["friction"] += friction_force
    pair["normal_load"] += max(0.0, normal_load)
    pair["friction_load"] += float(np.linalg.norm(friction_force))
    pair["max_depth"] = max(pair["max_depth"], depth)
    pair["depth_sum"] += depth
    pair["position_sum"] += position_world
    pair["negative_normal_count"] += normal_load < -1e-8


def _row(time_s: float, pair_name: str, pair: dict, poses: dict) -> dict:
    count = pair["count"]
    row = {
        "time_s": time_s,
        "pair": pair_name,
        "contact_count": count,
        "normal_load_sum_n": pair["normal_load"],
        "friction_load_sum_n": pair["friction_load"],
        "penetration_max_mm": pair["max_depth"] * 1000,
        "penetration_mean_mm": pair["depth_sum"] * 1000 / count if count else 0.0,
        "negative_normal_count": pair["negative_normal_count"],
    }
    for name, vector in (
        ("total_force", pair["force"]),
        ("normal_force", pair["normal"]),
        ("friction_force", pair["friction"]),
    ):
        for axis, value in zip("xyz", vector, strict=True):
            row[f"{name}_{axis}_n"] = float(value)
        row[f"{name}_magnitude_n"] = float(np.linalg.norm(vector))
    center = pair["position_sum"] / count if count else np.zeros(3)
    for axis, value in zip("xyz", center, strict=True):
        row[f"contact_center_{axis}_m"] = float(value)
    for name, vector in poses.items():
        for axis, value in zip("xyz", vector[:3], strict=True):
            row[f"{name}_{axis}_m"] = float(value)
        if len(vector) == 7:
            for axis, value in zip("wxyz", vector[3:], strict=True):
                row[f"{name}_quat_{axis}"] = float(value)
    return row


def _save_csv(path: Path, samples: list[tuple[float, dict, dict]]) -> None:
    if not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = _row(samples[0][0], PAIRS[0], samples[0][1][PAIRS[0]], samples[0][2])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        for time_s, pairs, poses in samples:
            for name in PAIRS:
                writer.writerow(_row(time_s, name, pairs[name], poses))


def _event_sample(time_s: float, step: int, dt: float) -> bool:
    if not 5.0 <= time_s <= 9.0:
        return False
    if 7.2 <= time_s <= 7.8:
        return step % round(0.002 / dt) == 0
    return step % round(0.01 / dt) == 0


def _surface_from_meshes(fruit_mesh, stem_mesh, local_points: np.ndarray) -> list[str]:
    import trimesh

    if not len(local_points):
        return []
    _, fruit_distance, _ = trimesh.proximity.closest_point(fruit_mesh, local_points)
    _, stem_distance, _ = trimesh.proximity.closest_point(stem_mesh, local_points)
    return [
        "fruit" if fruit <= stem + 0.0002 else "stem"
        for fruit, stem in zip(fruit_distance, stem_distance, strict=True)
    ]


def _run_superdex(seconds: float, output: Path) -> dict:
    os.environ["SUPERDEX_PRECISION"] = "fp64"
    sys.path.insert(0, str(ROOT / "superdex"))
    import native
    import run as task_run
    from superdex import physics

    if not physics.uses_double_precision():
        raise RuntimeError("SuperDex FP64 runtime is unavailable")
    import trimesh

    fruit_mesh = trimesh.load_mesh(
        ROOT / "superdex/assets/objects/apple-collision.obj", process=False
    )
    stem_mesh = trimesh.load_mesh(
        ROOT / "superdex/assets/objects/stem-collision.obj", process=False
    )
    coarse: list[tuple[float, dict, dict]] = []
    event: list[tuple[float, dict, dict]] = []
    query_checks: list[dict] = []
    original_dynamics = native.Instrumentation.dynamics

    def sample(instrumentation, time_s: float, *, verify_query=False) -> tuple[float, dict, dict]:
        apple = instrumentation.apple
        apple_handle = apple.get_handle()
        table_handle = instrumentation.table.get_handle()
        links = dict(zip(instrumentation.names, instrumentation.links, strict=True))
        handles = {link.get_handle().value: name for name, link in links.items()}
        transform = apple.get_root_transform()
        position = np.asarray(transform.translation, dtype=float)
        quaternion = np.asarray(transform.rotation, dtype=float)
        orientation = Rotation.from_quat(quaternion)
        poses = {
            "apple": np.r_[position, quaternion[[3, 0, 1, 2]]],
        }
        for short, name in (
            ("wrist", "r_wrist"),
            ("index_pad", "r_index_finger_pad"),
            ("thumb_pad", "r_thumb_pad"),
        ):
            part = links[name].get_root_transform()
            poses[short] = np.asarray(part.translation, dtype=float)
        pairs = {name: _empty_pair() for name in PAIRS}
        candidates = []
        for point in apple.get_contact_points_world():
            apple_is_a = point.actor_a == apple_handle
            other = point.actor_b if apple_is_a else point.actor_a
            force = np.asarray(point.force, dtype=float) * (1 if apple_is_a else -1)
            normal = np.asarray(point.normal, dtype=float) * (1 if apple_is_a else -1)
            contact_position = np.asarray(
                point.pos_a if apple_is_a else point.pos_b, dtype=float
            )
            if other == table_handle:
                name = "apple_table"
            else:
                link_name = handles.get(other.value)
                if link_name == "r_index_finger_pad":
                    name = "index"
                elif link_name == "r_thumb_pad":
                    name = "thumb"
                else:
                    name = "other_apple_contact"
            candidates.append((name, force, normal, contact_position, float(point.distance)))
        pad_points = np.asarray(
            [orientation.inv().apply(pos - position) for name, _, _, pos, _ in candidates
             if name in ("index", "thumb")],
            dtype=float,
        ).reshape(-1, 3)
        surfaces = iter(_surface_from_meshes(fruit_mesh, stem_mesh, pad_points))
        for name, force, normal, contact_position, distance in candidates:
            if name in ("index", "thumb"):
                name = f"{name}_{next(surfaces)}"
            if np.linalg.norm(force) <= 1e-12:
                continue
            signed_load = float(force @ normal)
            _add_contact(
                pairs[name], force, signed_load * normal, signed_load,
                max(0.0, -distance), contact_position,
            )
        if verify_query:
            for label, link in (
                ("index", links["r_index_finger_pad"]),
                ("thumb", links["r_thumb_pad"]),
                ("table", instrumentation.table),
            ):
                query = np.asarray(apple.get_contact_force_from_actor_world(link), dtype=float)
                summed = (
                    pairs[f"{label}_stem"]["force"] + pairs[f"{label}_fruit"]["force"]
                    if label != "table" else pairs["apple_table"]["force"]
                )
                query_checks.append({"time_s": time_s, "pair": label,
                                     "point_sum_minus_query_n": (summed - query).tolist()})
        return time_s, pairs, poses

    def recorded_dynamics(self, time_s, velocity_before):
        row = original_dynamics(self, time_s, velocity_before)
        completed = round(time_s / task_run.DT) + 1
        sample_time = completed * task_run.DT
        if completed % round(SAMPLE_DT / task_run.DT) == 0:
            coarse.append(sample(self, sample_time, verify_query=True))
        if _event_sample(sample_time, completed, task_run.DT):
            event.append(sample(self, sample_time))
        return row

    native.Instrumentation.dynamics = recorded_dynamics
    try:
        with tempfile.TemporaryDirectory(prefix="unisim-apple-compare-") as directory:
            robot = task_run.materialize_robot(Path(directory))
            backend = task_run.make_backend(robot)
            try:
                summary = task_run.run_task(
                    backend, seconds, robot=robot, output=output / "original"
                )
            finally:
                backend.close()
    finally:
        native.Instrumentation.dynamics = original_dynamics
    _save_csv(output / "coarse.csv", coarse)
    _save_csv(output / "event.csv", event)
    (output / "query_checks.json").write_text(json.dumps(query_checks, indent=2) + "\n")
    return {
        "engine": "superdex", "physics_dt_s": task_run.DT,
        "steps": summary["physics_steps"], "warnings": [],
        "apple_initial_position_m": coarse[0][2]["apple"][:3].tolist(),
        "source_robot_sha256": _sha256(task_run.ROBOT),
        "source_objects_sha256": _sha256(task_run.OBJECTS),
        "summary": summary,
    }


def _run_mujoco(seconds: float, output: Path) -> dict:
    import mujoco
    import trimesh

    sys.path.insert(0, str(ROOT / "mujoco_src"))
    import planning
    import runtime

    fd, name = tempfile.mkstemp(prefix="runtime-contact-compare-", suffix=".xml",
                                 dir=ROOT / "mujoco")
    os.close(fd)
    scene = Path(name)
    try:
        runtime.write_runtime_scene(path=scene)
        model, collision = runtime.compile_physics_model(scene)
    finally:
        scene.unlink(missing_ok=True)
    names = runtime.hinge_names(model)
    kin = planning.Kinematics()
    if names != kin.names:
        raise RuntimeError("Planner and loaded MuJoCo joint order differ")
    kp, kv = runtime.build_position_gains(names)
    runtime._apply_model_gains(model, names, kp, kv)
    qadr, _ = runtime.model_joint_arrays(model)
    apple_joint = model.joint("apple_with_stem_root")
    apple_qadr = int(apple_joint.qposadr[0])
    apple_geom = int(model.geom("apple_with_stem_collision").id)
    apple_body = int(model.body("apple_with_stem").id)
    table_geom = int(model.geom("table").id)
    pad_geoms = {
        int(model.geom("r_index_finger_pad_collision").id): "index",
        int(model.geom("r_thumb_pad_collision").id): "thumb",
    }
    fruit_mesh = trimesh.load_mesh(
        ROOT / "superdex/assets/objects/apple-collision.obj", process=False
    )
    stem_mesh = trimesh.load_mesh(
        ROOT / "superdex/assets/objects/stem-collision.obj", process=False
    )
    apple_pose = model.qpos0[apple_qadr : apple_qadr + 7].copy()
    plan = planning.GraspPlan(
        kin, apple_pose, json.loads((ROOT / "superdex/assets/stem.json").read_text())
    )
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[qadr] = plan.pre
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pose_data = mujoco.MjData(model)
    body_ids = {
        "wrist": int(model.body("r_wrist").id),
        "index_pad": int(model.body("r_index_finger_pad").id),
        "thumb_pad": int(model.body("r_thumb_pad").id),
    }
    local_force = np.zeros(6)

    def sample(time_s: float) -> tuple[float, dict, dict]:
        # mj_step leaves contacts/forces at the step's dynamics state; qpos is
        # already advanced. Read body poses from a separate data instance.
        pose_data.qpos[:] = data.qpos
        pose_data.qvel[:] = data.qvel
        mujoco.mj_kinematics(model, pose_data)
        apple = data.qpos[apple_qadr : apple_qadr + 7].copy()
        poses = {"apple": apple}
        for short, body_id in body_ids.items():
            poses[short] = np.asarray(pose_data.xpos[body_id], dtype=float).copy()
        pairs = {name: _empty_pair() for name in PAIRS}
        rotation = np.asarray(data.xmat[apple_body]).reshape(3, 3)
        candidates = []
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == apple_geom:
                other, sign = contact.geom2, -1
            elif contact.geom2 == apple_geom:
                other, sign = contact.geom1, 1
            else:
                continue
            if other == table_geom:
                label = "apple_table"
            else:
                label = pad_geoms.get(other, "other_apple_contact")
            candidates.append((i, label, sign, np.asarray(contact.pos).copy(),
                               float(contact.dist)))
        pad_points = np.asarray(
            [(pos - data.xpos[apple_body]) @ rotation for _, label, _, pos, _ in candidates
             if label in ("index", "thumb")],
            dtype=float,
        ).reshape(-1, 3)
        surfaces = iter(_surface_from_meshes(fruit_mesh, stem_mesh, pad_points))
        for i, label, sign, position, distance in candidates:
            if label in ("index", "thumb"):
                label = f"{label}_{next(surfaces)}"
            local_force.fill(0.0)
            mujoco.mj_contactForce(model, data, i, local_force)
            if local_force[0] <= 0:
                continue
            frame = np.asarray(data.contact[i].frame).reshape(3, 3)
            normal = sign * frame[0] * local_force[0]
            friction = sign * (frame.T @ np.r_[0.0, local_force[1:3]])
            _add_contact(
                pairs[label], normal + friction, normal, float(local_force[0]),
                max(0.0, -distance), position,
            )
        return time_s, pairs, poses

    coarse: list[tuple[float, dict, dict]] = []
    event: list[tuple[float, dict, dict]] = []
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda warning: warnings.append(str(warning)))
    completed = 0
    try:
        for step in range(round(seconds / runtime.DT)):
            time_s = step * runtime.DT
            if step == round(runtime.GAIN_SWITCH_TIME / runtime.DT):
                plan.regrasp = planning.plan_body_grasp(
                    kin, data.qpos[qadr].copy(), data.qpos[apple_qadr : apple_qadr + 3].copy()
                )
                kp, kv = runtime.build_position_gains(names, body_grasp=True)
                runtime._apply_model_gains(model, names, kp, kv)
            data.ctrl[:] = plan.target(time_s)
            mujoco.mj_step(model, data)
            completed = step + 1
            if warnings or not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                break
            sampled_time = completed * runtime.DT
            if completed % round(SAMPLE_DT / runtime.DT) == 0:
                coarse.append(sample(sampled_time))
            if _event_sample(sampled_time, completed, runtime.DT):
                event.append(sample(sampled_time))
            if completed % round(1 / runtime.DT) == 0:
                print(f"MuJoCo {sampled_time:.1f} s, apple z={data.qpos[apple_qadr+2]:.5f}",
                      flush=True)
    finally:
        mujoco.set_mju_user_warning(old_warning)
    _save_csv(output / "coarse.csv", coarse)
    _save_csv(output / "event.csv", event)
    return {
        "engine": "mujoco", "physics_dt_s": runtime.DT,
        "steps": completed, "warnings": warnings,
        "apple_initial_position_m": apple_pose[:3].tolist(),
        "source_scene_sha256": _sha256(runtime.SCENE),
        "compiled_collision": collision,
        "pair_solref": model.pair_solref.tolist(),
        "pair_solimp": model.pair_solimp.tolist(),
        "pair_friction": model.pair_friction.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("mujoco", "superdex"), required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > 40:
        parser.error("--seconds must be finite and in (0, 40]")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "superdex":
        result = _run_superdex(args.seconds, args.output)
    else:
        result = _run_mujoco(args.seconds, args.output)
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "summary"},
                     indent=2))
    return 0 if result["steps"] == round(args.seconds / result["physics_dt_s"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
