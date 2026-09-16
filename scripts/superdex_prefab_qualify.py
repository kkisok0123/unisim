#!/usr/bin/env python3
"""Qualify FR3 with nested sphere and peg/board prefabs, or view the same scene."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from superdex_bot_qualify import INVENTORY_REPORT, _git, resolve_assets_root  # noqa: E402

from unisim import create_backend  # noqa: E402
from unisim.backend.superdex.assets import verify_asset_bundle  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402

DT = 0.001
LIMITS = np.array([87.0] * 4 + [12.0] * 3)
BOT = "bots/arms/fr3_v2/fr3_v2.superdex_bot"
SPHERE = "prefabs/sphere/sphere.mochi_prefab"
BOARD = "prefabs/nine_hole_peg_test/nine_hole_peg_test.mochi_prefab"


def fixture(root: Path, directory: Path) -> SceneCfg:
    """Generate wrapper files only; preserve all original asset bytes."""
    child = directory / "objects.mochi_prefab"
    child.write_text(
        json.dumps(
            {
                "prefabs": [
                    {"name": "probe", "path": str(root / SPHERE), "translation": [-0.09, 0, 0.23]},
                    {"name": "board", "path": str(root / BOARD), "translation": [0.55, 0, 0]},
                ]
            }
        )
    )
    assembly = directory / "assembly.mochi_prefab"
    assembly.write_text(
        json.dumps(
            {
                "prefabs": [
                    {
                        "name": "assembly",
                        "path": "./objects.mochi_prefab",
                        "translation": [0, 0, 2.532],
                        "rotation": [0, 0, np.sin(0.2), np.cos(0.2)],
                    }
                ]
            }
        )
    )
    return SceneCfg(str(root / BOT), fragment_files=[str(assembly)])


def backend(scene: SceneCfg, mode: str, count: int = 2):
    return create_backend(
        "superdex",
        scene,
        count,
        DT,
        superdex_execution_mode=mode,
        superdex_effort_limits=LIMITS,
    )


def control(q, v, target):
    return np.clip(80 * (target - q[..., :7]) - 8 * v[..., :7], -LIMITS, LIMITS)


def qualify(root: Path, directory: Path, steps: int = 1000) -> dict:
    import superdex.physics as p
    import superdex.robotics as r

    scene_cfg = fixture(root, directory)
    result = {}
    for mode in ("serial", "batch"):
        b = backend(scene_cfg, mode)
        direct = None
        bot = None
        try:
            direct = p.create_scene("stage5_direct_reference")
            direct.set_gravity([0, 0, -9.81])
            owner = r.create_context()
            cfg = r.load_bot_prefab_from_file(scene_cfg.model_file)
            bot = r.create_bot(direct, cfg, owner)
            robot = bot.get_articulated_actor()
            loaded = p.prefab.load_from_file(scene_cfg.fragment_files[0], str(root))
            objects = list(p.prefab.add_to_scene(loaded, direct).actors)
            assert [a.get_name() for a in objects] == [item.name for item in b.model.rigids]
            assert len(objects) == 11  # One sphere, one static board, nine dynamic pegs.
            assert sum(a.is_static() for a in objects) == 1
            assert direct.get_num_actors() == b._worlds[0].get_num_actors()
            inventory = []
            for actor in objects:
                pose = actor.get_root_transform()
                inventory.append(
                    {
                        "name": actor.get_name(),
                        "static": actor.is_static(),
                        "collision_shape_hash": str(actor.get_reference_shape().get_hash()),
                        "initial_world_xyz": np.asarray(pose.translation).tolist(),
                        "initial_world_wxyz": np.asarray(pose.rotation)[[3, 0, 1, 2]].tolist(),
                        "mass": None if actor.is_static() else float(actor.get_mass()),
                    }
                )
            robot_handles = set(robot.get_nested_link_actors())
            probe = next(a for a in objects if a.get_name().endswith("/Sphere"))
            probe_index = next(i for i, a in enumerate(objects) if a == probe)
            adapter_handles = set(b._actors[0].get_nested_link_actors())
            probe.register_query(p.QueryType.CONTACT_POINTS)
            b._rigids[0][probe_index].register_query(p.QueryType.CONTACT_POINTS)
            initial = b.get_state()
            target = initial["qpos"][0, :7].copy()
            dtype = initial["qpos"].dtype
            q, v = np.empty(7, dtype=dtype), np.empty(7, dtype=dtype)
            peak_error = 0.0
            contact_steps = 0
            for _ in range(steps):
                state = b.get_state()
                commands = control(state["qpos"], state["qvel"], target)
                robot.get_articulated_pose(q)
                robot.get_articulated_joint_velocities(v)
                robot.set_external_forces_on_dofs(
                    np.arange(7, dtype=np.int32), control(q, v, target).astype(dtype)
                )
                b.step(commands)
                direct.step(DT)
                assert direct.get_solver_stats().convergence_status != p.ConvergenceStatus.DIVERGED
                robot.get_articulated_pose(q)
                robot.get_articulated_joint_velocities(v)
                current = b.get_state()
                deviations = [
                    np.max(abs(current["qpos"][0, :7] - q)),
                    np.max(abs(current["qvel"][0, :7] - v)),
                ]
                for item, actor in zip(b.model.rigids, objects, strict=True):
                    body = np.array([item.body_id])
                    pose = actor.get_root_transform()
                    deviations.extend(
                        [
                            np.max(abs(b.get_body_pos_w(body)[0, 0] - pose.translation)),
                            np.max(
                                abs(
                                    b.get_body_quat_w(body)[0, 0]
                                    - np.asarray(pose.rotation)[[3, 0, 1, 2]]
                                )
                            ),
                            np.max(
                                abs(b.get_body_ang_vel_w(body)[0, 0] - actor.get_angular_velocity())
                            ),
                        ]
                    )
                peak_error = max(peak_error, float(max(deviations)))
                points = probe.get_contact_points_world()
                found = any(
                    pt.distance <= 1e-4
                    and (pt.actor_a in robot_handles or pt.actor_b in robot_handles)
                    for pt in points
                )
                adapter_found = any(
                    pt.distance <= 1e-4
                    and (pt.actor_a in adapter_handles or pt.actor_b in adapter_handles)
                    for pt in b._rigids[0][probe_index].get_contact_points_world()
                )
                assert found == adapter_found
                if found:
                    assert adapter_found
                    contact_steps += 1
            assert peak_error < 3e-5, peak_error
            assert contact_steps > 0, "sphere never contacted the robot"
            moved = b.get_state()
            assert np.max(abs(moved["qpos"][:, 7:] - initial["qpos"][:, 7:])) > 0.01
            b.reset(np.array([0]))
            for key in ("qpos", "qvel"):
                np.testing.assert_allclose(b.get_state()[key][0], initial[key][0], atol=2e-6)
                np.testing.assert_array_equal(b.get_state()[key][1], moved[key][1])
            # Give every dynamic object a nontrivial pose and twist, including COM offsets.
            roundtrip = {key: value[[0]].copy() for key, value in moved.items()}
            for item in b.model.rigids:
                if item.qpos_index is None:
                    continue
                qi, vi = item.qpos_index, item.qvel_index
                roundtrip["qpos"][0, qi : qi + 3] += [0.2, -0.1, 0.3]
                roundtrip["qpos"][0, qi + 3 : qi + 7] = [np.cos(0.2), np.sin(0.2), 0, 0]
                roundtrip["qvel"][0, vi : vi + 6] = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
            b.set_state(np.array([0]), roundtrip["qpos"], roundtrip["qvel"])
            for key in ("qpos", "qvel"):
                np.testing.assert_allclose(b.get_state()[key][0], roundtrip[key][0], atol=3e-6)
                np.testing.assert_array_equal(b.get_state()[key][1], moved[key][1])
            # A selected scene can advance independently; refresh checks no cross-scene ownership.
            unchanged = b.get_state()
            b._worlds[0].step(DT)
            b._refresh(np.array([0]))
            for key in unchanged:
                np.testing.assert_array_equal(b.get_state()[key][1], unchanged[key][1])
            b.reset()
            for key in initial:
                np.testing.assert_allclose(b.get_state()[key], initial[key], atol=2e-6)
            result[mode] = {
                "steps": steps,
                "max_adapter_sdk_error": peak_error,
                "robot_object_contact_steps": contact_steps,
                "scene_actor_count": direct.get_num_actors(),
                "rigid_bodies": [item.name for item in b.model.rigids],
                "rigid_inventory": inventory,
                "nq": b.model.nq,
                "nv": b.model.nv,
                "whole_scene_reset": "passed",
                "selective_reset": "passed",
                "state_roundtrip": "passed",
                "environment_isolation": "passed",
            }
        finally:
            if bot is not None:
                r.destroy_bot(direct, bot)
            if direct is not None:
                p.destroy_scene(direct)
            b.close()
            b.close()
        reopened = backend(scene_cfg, mode)
        reopened.close()
        result[mode]["cleanup_recreation"] = "passed"
    return result


def view(root: Path, directory: Path, frames: int | None) -> dict:
    b = backend(fixture(root, directory), "serial", 1)
    target = b.get_default_qpos()[:7]
    frame = 0
    reset_error = None

    def advance(obs):
        nonlocal frame, reset_error
        if frame == 0:
            print("Initial pose: inspect robot, sphere and peg board.")
        if frame == 120:
            print("Contact: sphere falls onto the robot; pegs settle in the board.")
        if 120 <= frame < 360:
            state = b.get_state()
            b.step(control(state["qpos"], state["qvel"], target), nsteps=8)
        if frame == 360:
            b.reset()
            error = float(np.max(abs(b.get_state()["qpos"][0] - b.get_default_qpos())))
            reset_error = error
            print(f"Whole-scene reset: max coordinate error {error:.3g}; inspect restored objects.")
        frame += 1
        return obs

    try:
        b.run_playback(
            env=SimpleNamespace(cfg=SimpleNamespace(ctrl_dt=1 / 60)),
            initialize=lambda: None,
            step=advance,
            num_steps=frames,
            headless=frames is not None,
            record_video=False,
        )
    finally:
        b.close()
    if frame <= 360:
        raise RuntimeError("viewer closed before the reset phase")
    return {
        "frames": frame,
        "mode": "offscreen" if frames is not None else "interactive",
        "reset_max_error": reset_error,
        "cleanup": "passed",
        "phases": ["initial-pose", "contact", "whole-scene-reset"],
        "visual_inspection": "not claimed; automated native renderer smoke only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--frames", type=int, help="offscreen viewer frames; at least 361")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = resolve_assets_root(args.assets)
    verified = verify_asset_bundle(root, INVENTORY_REPORT)
    with tempfile.TemporaryDirectory(prefix="unisim-prefabs-") as temporary:
        directory = Path(temporary)
        if args.frames is not None and (not args.viewer or args.frames < 361):
            parser.error("--frames requires --viewer and at least 361 frames")
        import superdex.physics as physics

        report = {
            "code_commit": _git("rev-parse", "HEAD"),
            "working_tree": _git("status", "--short"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sdk": importlib.metadata.version("superdex-physics-uni"),
            "manifest_sha256": hashlib.sha256(INVENTORY_REPORT.read_bytes()).hexdigest(),
            "asset_verification": verified,
            "asset_provenance": json.loads(INVENTORY_REPORT.read_text())["provenance"],
            "precision": "float64" if physics.uses_double_precision() else "float32",
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    *sorted((ROOT / "src/unisim/backend/superdex").glob("*.py")),
                    Path(__file__).resolve(),
                ]
            },
            "assets": [BOT, SPHERE, BOARD],
            "dt": DT,
            "control": "PD effort: kp=80, kd=8; authored robot pose target",
            "effort_limits": LIMITS.tolist(),
            "tolerance": 3e-5,
            "results": (
                view(root, directory, args.frames) if args.viewer else qualify(root, directory)
            ),
            "visual_verification": "native renderer smoke only" if args.viewer else "not performed",
        }
        if args.out is None:
            filename = "viewer.json" if args.viewer else "report.json"
            args.out = ROOT / "docs/superdex-prefab-qualification" / filename
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
