#!/usr/bin/env python3
"""Qualify native floating hands against independent direct SDK scenes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from superdex_bot_qualify import (  # noqa: E402
    INVENTORY_REPORT,
    _git,
    _load_prefab,
    resolve_assets_root,
)

from unisim import create_backend  # noqa: E402
from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402

HANDS = {
    "allegro_v5_right": "bots/hands/allegro_v5/right/allegro_v5_right.superdex_bot",
    "dg5f_short_left": "bots/hands/dg5f_short/left/dg5f_short_left.superdex_bot",
}
DT = 0.002


def discover_floating(root: Path) -> dict[str, str]:
    """Discover compiled FREE roots, including recipes, from the local bundle."""
    import superdex.physics as physics

    found = {}
    for path in sorted(root.rglob("*.superdex_bot")):
        cfg = _load_prefab(path)
        if cfg.joints[0].type == physics.ArticulatedJointType.FREE:
            if path.stem in found:
                raise ValueError(f"duplicate native bot key: {path.stem}")
            found[path.stem] = path.relative_to(root).as_posix()
    return found


def rotation_matrix(xyzw):
    """Independent quaternion matrix for the direct SDK reference."""
    x, y, z, w = np.asarray(xyzw, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def qualify(path: Path, steps: int = 1040) -> dict:
    """Fail on any frame, trajectory, reset, isolation or ownership regression."""
    import superdex.physics as p
    import superdex.robotics as r

    cfg = r.load_bot_prefab_from_file(str(path))
    n = sum(
        j.type in (p.ArticulatedJointType.REVOLUTE, p.ArticulatedJointType.PRISMATIC)
        for j in cfg.joints
    )
    limits = np.array(
        [
            float(j.effort_limit)
            for j in list(cfg.joints)[1:]
            if j.type != p.ArticulatedJointType.HARD
        ]
    )
    limits = np.where(limits > 0, limits, 1.0)
    results = {}
    for mode in ("serial", "batch"):
        backend = create_backend(
            "superdex",
            SceneCfg(str(path)),
            2,
            DT,
            superdex_execution_mode=mode,
            superdex_effort_limits=limits,
        )
        scene = None
        bot = None
        try:
            scene = p.create_scene("floating_reference")
            scene.set_gravity([0, 0, -9.81])
            owner = r.create_context()
            bot = r.create_bot(scene, cfg, owner)
            actor = bot.get_articulated_actor()
            initial_snapshot = scene.capture_state()
            links = [scene.get_actor(h) for h in actor.get_nested_link_actors()]
            dtype = np.float64 if p.uses_double_precision() else np.float32
            native_q, native_v = np.empty(n + 6, dtype=dtype), np.empty(n + 6, dtype=dtype)
            actor.get_articulated_pose(native_q)
            default = backend.get_default_qpos()
            assert default.shape == (n + 7,)
            assert backend.get_state()["qvel"].shape == (2, n + 6)
            assert list(backend.model.body_names) == [
                "world",
                *[str(link.name) for link in cfg.links],
            ]
            np.testing.assert_allclose(default[7:], native_q[6:], atol=1e-7)
            np.testing.assert_allclose(default[:3], links[0].get_root_transform().translation)
            np.testing.assert_allclose(
                default[3:7], np.asarray(links[0].get_root_transform().rotation)[[3, 0, 1, 2]]
            )
            np.testing.assert_array_equal(backend.model.actuator_qpos_indices, np.arange(n) + 7)
            np.testing.assert_array_equal(backend.model.actuator_qvel_indices, np.arange(n) + 6)
            root_layout = backend.get_root_state_layout(str(cfg.links[0].name))
            assert root_layout.qpos_indices == (0, 1, 2, 3, 4, 5, 6)
            assert root_layout.qvel_indices == (0, 1, 2, 3, 4, 5)
            for index, link in enumerate(links):
                body = index + 1
                pose = link.get_root_transform()
                np.testing.assert_allclose(
                    backend.get_body_pos_w(np.array([body]))[0, 0], pose.translation, atol=2e-6,
                )
                np.testing.assert_allclose(
                    backend.get_body_mass()[body], link.get_mass(), atol=1e-6
                )
                actual_box = backend._links[0][index].get_aabb_world()
                reference_box = link.get_aabb_world()
                np.testing.assert_allclose([actual_box.min, actual_box.max],
                                           [reference_box.min, reference_box.max], atol=2e-6)
            # Nonidentity orientation and nonparallel velocities expose frame mistakes.
            q = default.copy()
            q[:3] = [0.2, -0.3, 1.5]
            quat = p.Quaternion.from_rotation_vector(np.array([0.4, -0.2, 0.7], dtype=dtype))
            q[3:7] = np.asarray(quat)[[3, 0, 1, 2]]
            v = np.linspace(-0.08, 0.12, n + 6).astype(dtype)
            rotation = rotation_matrix(quat)
            backend.set_state(np.array([0]), q[None], v[None])
            np.testing.assert_allclose(backend.get_state()["qpos"][0], q, atol=1e-6)
            np.testing.assert_allclose(backend.get_state()["qvel"][0], v, atol=1e-6)
            # Use SDK transform composition and the SDK root-link Jacobian as
            # the independent reference, not the adapter's frame helpers.
            world_pose = p.TransformRT(translation=q[:3], rotation=quat)
            native_pose = (cfg.joints[0].parent_link_from_joint.inverse() * world_pose
                           * cfg.links[0].parent_joint_from_link.inverse())
            native_q[:3] = np.asarray(native_pose.translation)
            native_q[3:6] = np.asarray(native_pose.rotation.to_rotation_vector())
            native_q[6:] = q[7:]
            actor.set_articulated_pose_from_joints(native_q)
            scene.step(0)
            root_pose = links[0].get_root_transform()
            com_offset = (np.asarray(links[0].get_center_of_mass_transform().translation)
                          - np.asarray(root_pose.translation))
            angular = rotation @ v[3:6]
            jacobian = np.asarray(links[0].get_articulated_jacobian()).reshape(6, n + 6)
            native_v[:6] = np.linalg.solve(
                jacobian[:, :6], np.r_[v[:3] + np.cross(angular, com_offset), angular]
            )
            native_v[6:] = v[6:]
            # Validate the mapping independently, then use identical float32
            # native inputs for dynamics. Tiny quaternion-composition rounding
            # differences otherwise select different contact/friction branches.
            actual_q, actual_v = np.empty_like(native_q), np.empty_like(native_v)
            backend._actors[0].get_articulated_pose(actual_q)
            backend._actors[0].get_articulated_joint_velocities(actual_v)
            np.testing.assert_allclose(actual_q, native_q, atol=2e-6, rtol=1e-6)
            np.testing.assert_allclose(actual_v, native_v, atol=2e-6, rtol=1e-6)
            scene.restore_state(initial_snapshot, release_immediately=True)
            backend._worlds[0].restore_state(backend._snapshots[0], release_immediately=False)
            backend._actors[0].set_articulated_pose_from_joints(native_q)
            backend._actors[0].set_articulated_joint_velocities(native_v)
            backend._worlds[0].step(0)
            backend._refresh(np.array([0]))
            actor.set_articulated_pose_from_joints(native_q)
            actor.set_articulated_joint_velocities(native_v)
            scene.step(0)
            max_error = 0.0
            initial = backend.get_state()["qpos"][0].copy()
            for k in range(steps):
                torque = (0.001 * np.sin(k * DT * 3 + np.arange(n))).astype(dtype)
                backend.step(np.stack((torque, np.zeros(n))), 1)
                force = np.zeros(n + 6, dtype=dtype)
                force[6:] = torque
                actor.set_external_forces_on_dofs(np.arange(n + 6, dtype=np.int32), force)
                scene.step(DT)
                actor.get_articulated_pose(native_q)
                actor.get_articulated_joint_velocities(native_v)
                state = backend.get_state()
                pose = links[0].get_root_transform()
                pos = np.asarray(pose.translation)
                ang = np.asarray(links[0].get_angular_velocity())
                origin_vel = np.asarray(links[0].get_linear_velocity()) - np.cross(
                    ang, np.asarray(links[0].get_center_of_mass_transform().translation) - pos
                )
                quat_ref = np.asarray(pose.rotation)[[3, 0, 1, 2]]
                qref = np.r_[pos, quat_ref, native_q[6:]]
                if np.dot(qref[3:7], state["qpos"][0, 3:7]) < 0:
                    qref[3:7] *= -1
                rot = rotation_matrix(pose.rotation)
                # SDK link velocity readouts lose precision during long free
                # fall. Its Jacobian maps generalized velocity to the COM twist.
                jacobian = np.asarray(links[0].get_articulated_jacobian()).reshape(6, n + 6)
                twist = jacobian @ native_v
                reference_linear = twist[:3] - np.cross(
                    twist[3:], np.asarray(links[0].get_center_of_mass_transform().translation) - pos
                )
                vref = np.r_[reference_linear, rot.T @ twist[3:], native_v[6:]]
                if k == 0:
                    np.testing.assert_allclose(origin_vel, reference_linear, atol=1e-4)
                    np.testing.assert_allclose(ang, twist[3:], atol=1e-4)
                error = max(
                    float(np.max(np.abs(state["qpos"][0] - qref))),
                    float(np.max(np.abs(state["qvel"][0] - vref))),
                )
                max_error = max(max_error, error)
                np.testing.assert_allclose(state["qpos"][0], qref, atol=2e-6, rtol=1e-6)
                np.testing.assert_allclose(state["qvel"][0], vref, atol=2e-6, rtol=1e-6)
            assert np.linalg.norm(state["qpos"][0, :3] - initial[:3]) > 0.01
            assert np.max(np.abs(state["qpos"][0, 7:] - initial[7:])) > 1e-5
            before = backend.get_state()
            backend.reset(np.array([0]))
            after = backend.get_state()
            for key in ("qpos", "qvel"):
                np.testing.assert_array_equal(after[key][1], before[key][1])
            np.testing.assert_allclose(after["qpos"][0], default, atol=1e-6)
            np.testing.assert_array_equal(after["qvel"][0], 0)
            # Environment 1 must match a separate scene given its own zero inputs.
            single = create_backend(
                "superdex",
                SceneCfg(str(path)),
                1,
                DT,
                superdex_execution_mode=mode,
                superdex_effort_limits=limits,
            )
            try:
                single.step(np.zeros((1, n)), steps)
                for key in ("qpos", "qvel"):
                    np.testing.assert_array_equal(before[key][1], single.get_state()[key][0])
            finally:
                single.close()
            backend.reset()
            np.testing.assert_allclose(
                backend.get_state()["qpos"], np.tile(default, (2, 1)), atol=1e-6
            )
            np.testing.assert_array_equal(backend.get_state()["qvel"], 0)
            # Saturating positive and negative commands must equal commands at
            # the explicit effort limits, including after a full reset.
            for sign in (-1, 1):
                backend.reset()
                backend.step(np.stack((sign * 2 * limits, sign * limits)), 1)
                clipped = backend.get_state()
                for field in ("qpos", "qvel"):
                    np.testing.assert_array_equal(clipped[field][0], clipped[field][1])
            results[mode] = {
                "steps": steps,
                "max_state_error": max_error,
                "roundtrip_reset_isolation": "passed",
                "body_geometry_inventory": "passed",
                "effort_clipping": "passed",
                "joints": n,
                "bodies": len(links),
                "effort_limits": limits.tolist(),
            }
        finally:
            if bot is not None:
                r.destroy_bot(scene, bot)
            if scene is not None:
                p.destroy_scene(scene)
            backend.close()
            backend.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets")
    parser.add_argument("--bots", default=",".join(HANDS),
                        help="comma-separated FREE-root bot filenames without extensions")
    parser.add_argument("--all-floating", action="store_true",
                        help="check every compiled FREE-root bot in the verified bundle")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/superdex-floating-qualification")
    args = parser.parse_args()
    import superdex.physics as physics

    root = resolve_assets_root(args.assets)
    verified = verify_asset_bundle(root, INVENTORY_REPORT)
    args.out.mkdir(parents=True, exist_ok=True)
    candidates = discover_floating(root)
    keys = list(candidates) if args.all_floating else args.bots.split(",")
    unknown = set(keys) - candidates.keys()
    if unknown:
        parser.error(f"unknown bot keys: {sorted(unknown)}")
    summary = []
    failed = False
    for key in keys:
        result = {}
        error = None
        status = "passed"
        try:
            result = qualify(root / candidates[key])
        except NotImplementedError as exc:
            status = "blocked"
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            # Numerical/validation failures must not be relabeled as unsupported.
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            failed = True
        report = {
            "bot": candidates[key],
            "status": status,
            "error": error,
            "checks": result,
            "asset_verification": verified,
            "code_commit": _git("rev-parse", "HEAD"),
            "working_tree": _git("status", "--short"),
            "diff_sha256": hashlib.sha256(_git("diff").encode()).hexdigest(),
            "asset_provenance": json.loads(INVENTORY_REPORT.read_text())["provenance"],
            "sdk_version": importlib.metadata.version("superdex-physics-uni"),
            "sdk_precision": "float64" if physics.uses_double_precision() else "float32",
            "code_file_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(__file__),
                    ROOT / "src/unisim/backend/superdex/materialization.py",
                    ROOT / "src/unisim/backend/superdex/backend.py",
                    ROOT / "src/unisim/backend/superdex/root_state.py",
                ]
            },
            "tolerance": {"state_atol": 2e-6, "state_rtol": 1e-6},
            "platform": platform.platform(),
            "python": platform.python_version(),
            "dt": DT,
            "gravity": [0, 0, -9.81],
            "visual_verification": "not performed by this headless command",
        }
        (args.out / f"{key}.json").write_text(json.dumps(report, indent=2) + "\n")
        summary.append({"key": key, "bot": candidates[key], "status": status, "error": error})
        print(key, status, error or result, flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Floating-model compatibility", "",
             "Generated by `superdex_floating_qualify.py --all-floating`. Passed means",
             "numerical state/lifecycle qualification in serial and batch modes;",
             "visual verification and object contact are separate.", "",
             "| Bot | Result | Evidence |", "| --- | --- | --- |"]
    for item in summary:
        evidence = (item["error"] or "1,040 steps per mode; state, reset and isolation passed")
        lines.append(f"| `{item['bot']}` | {item['status']} | "
                     f"[{evidence.replace('|', '/').replace(chr(10), ' ')}]({item['key']}.json) |")
    (args.out / "compatibility-table.md").write_text("\n".join(lines) + "\n")
    if failed or (not args.all_floating and any(item["status"] != "passed" for item in summary)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
